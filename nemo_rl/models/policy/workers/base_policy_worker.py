# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Optional

import ray
import torch
import zmq

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy.interfaces import ReferenceLogprobOutputSpec
from nemo_rl.utils.nsys import wrap_with_nvtx_name


def _recordable_sparse_delta_names(
    tracker: Any,
    update_params: list[tuple[int, str, torch.Tensor]],
) -> list[str] | None:
    if tracker is None or not tracker.is_delta_sync():
        return None

    names = []
    for _, name, param in update_params:
        baseline = tracker.baseline.get(name)
        if (
            baseline is None
            or tuple(baseline.shape) != tuple(param.shape)
            or baseline.dtype != param.dtype
        ):
            return None
        names.append(name)
    return names


class AbstractPolicyWorker:
    """Base class for policy workers with shared functionality."""

    def _setup_model_update_group(
        self, *, master_address: str, port: int, rank: int, world_size: int
    ) -> None:
        """Build the refit process group, init NCCL, and prewarm the baseline."""
        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup

        self.model_update_group = StatelessProcessGroup(
            master_address=master_address,
            port=port,
            rank=rank,
            world_size=world_size,
        )
        device = torch.cuda.current_device()
        self.model_update_group.init_nccl_communicator(device=device)
        self.prewarm_refit_payload_source_baseline_from_metadata()

    def init_collective(
        self, ip: str, port: int, world_size: int, *, train_world_size: int
    ) -> None:
        """Initialize the collective communication.

        Args:
            ip: IP address for the process group
            port: Port for the process group
            world_size: Total world size (train_world_size + inference_world_size)
            train_world_size: Number of training workers (used in inference cluster)
        """
        self._setup_model_update_group(
            master_address=ip, port=port, rank=self.rank, world_size=world_size
        )

    def is_refit_payload_source(self, *, fallback_to_rank: bool = False) -> bool:
        """Return whether this worker sources payloads for its refit group."""
        model_update_group = getattr(self, "model_update_group", None)
        if model_update_group is not None:
            return getattr(model_update_group, "rank", None) == 0
        return fallback_to_rank and getattr(self, "rank", None) == 0

    def prewarm_refit_payload_source_baseline_from_metadata(self) -> None:
        """Allocate delta-baseline storage on selected refit payload sources."""
        if not self.is_refit_payload_source():
            return
        tracker = getattr(self, "delta_weight_transfer_tracker", None)
        metadata = getattr(self, "_refit_param_info_for_delta_baseline", None)
        if tracker is None or metadata is None:
            return
        tracker.prewarm_baseline_from_metadata(metadata)

    def has_pending_full_sync_baseline(self) -> bool:
        """Return whether this worker still needs live full-sync baseline prewarm."""
        tracker = getattr(self, "delta_weight_transfer_tracker", None)
        if tracker is None:
            return False
        return tracker.has_pending_full_sync_baseline()

    @torch.no_grad()
    def apply_refit_benchmark_sparse_update(
        self,
        *,
        fraction: float,
        delta: float,
        seed: int,
        pattern: str = "contiguous",
    ) -> dict[str, float | int | str]:
        """Apply deterministic sparse updates for refit verifier benchmarks."""
        if fraction <= 0:
            return {
                "tensors": 0,
                "values": 0,
                "total_values": 0,
                "stride": 0,
                "pattern": "contiguous",
            }
        if fraction > 1:
            raise ValueError("fraction must be <= 1")
        if pattern not in {"contiguous", "strided"}:
            raise ValueError(f"Unsupported sparse update pattern: {pattern!r}")

        update_params = [
            (idx, name, param)
            for idx, (name, param) in enumerate(self.model.named_parameters())
            if param.requires_grad and param.dtype.is_floating_point
        ]
        tracker = getattr(self, "delta_weight_transfer_tracker", None)
        recorded_names = _recordable_sparse_delta_names(tracker, update_params)

        total_values = mutated_values = mutated_tensors = last_stride = 0
        rank_offset = int(self.rank) * 104729
        for update_idx, (param_idx, _param_name, param) in enumerate(update_params):
            flat = param.detach().view(-1)
            numel = int(flat.numel())
            total_values += numel
            update_values = min(numel, max(1, round(numel * fraction)))
            if pattern == "contiguous":
                start = (seed + rank_offset + param_idx * 1009) % (
                    numel - update_values + 1
                )
                flat.narrow(0, start, update_values).add_(delta)
                if recorded_names is not None and tracker is not None:
                    tracker.record_sparse_delta_range(
                        recorded_names[update_idx],
                        start=start,
                        values=torch.full(
                            (update_values,),
                            float(delta),
                            dtype=tracker.delta_dtype,
                            device="cpu",
                        ),
                    )
            else:
                last_stride = max(1, numel // update_values)
                start = (seed + rank_offset + param_idx * 1009) % last_stride
                locations = (
                    torch.arange(update_values, dtype=torch.long, device=flat.device)
                    .mul_(last_stride)
                    .add_(start)
                )
                deltas = torch.full(
                    (update_values,), float(delta), dtype=flat.dtype, device=flat.device
                )
                flat.index_add_(0, locations, deltas)
                if recorded_names is not None and tracker is not None:
                    tracker.record_sparse_delta_locations(
                        recorded_names[update_idx],
                        locations=locations,
                        values=deltas,
                    )
            mutated_tensors += 1
            mutated_values += update_values

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return {
            "tensors": mutated_tensors,
            "values": mutated_values,
            "total_values": total_values,
            "actual_fraction": mutated_values / total_values if total_values else 0.0,
            "stride": last_stride,
            "pattern": pattern,
        }

    def is_alive(self) -> bool:
        """Check if the worker is alive."""
        return True

    def reset_peak_memory_stats(self) -> None:
        """Reset peak memory statistics."""
        torch.cuda.reset_peak_memory_stats()

    def get_gpu_info(self) -> dict[str, Any]:
        """Return information about the GPU being used by this worker."""
        from nemo_rl.models.policy.utils import get_gpu_info

        return get_gpu_info(self.model)

    def report_device_id(self) -> str:
        """Report the UUID of the current CUDA device using NVML.

        Returns:
            str: UUID of the device in the format "GPU-xxxxx"
        """
        from nemo_rl.utils.nvml import get_device_uuid

        # Get current device index from torch
        device_idx = torch.cuda.current_device()
        # Get device UUID using NVML
        return get_device_uuid(device_idx)

    def get_zmq_address(self) -> str:
        """Get the ZMQ address for the current device."""
        return f"ipc:///tmp/{self.report_device_id()}.sock"

    def maybe_init_zmq(self) -> None:
        """Initialize the ZMQ socket if it doesn't exist."""
        if not hasattr(self, "zmq_socket"):
            self.zmq_context = zmq.Context()
            self.zmq_socket = self.zmq_context.socket(zmq.REQ)
            self.zmq_socket.setsockopt(
                zmq.SNDTIMEO, 120000
            )  # set timeout to 120 seconds
            self.zmq_socket.setsockopt(
                zmq.RCVTIMEO, 120000
            )  # set timeout to 120 seconds
            self.zmq_socket.setsockopt(zmq.LINGER, 0)
            self.zmq_socket.bind(self.get_zmq_address())

    def get_free_memory_bytes(self) -> int:
        """Get the available free memory."""
        from nemo_rl.utils.nvml import get_free_memory_bytes

        device_idx = torch.cuda.current_device()
        return get_free_memory_bytes(device_idx)

    def shutdown(self) -> bool:
        """Shutdown the policy."""
        try:
            # Clean up extension resources like ZMQ sockets
            if hasattr(self, "zmq_socket"):
                self.zmq_socket.close()
                self.zmq_context.term()
            return True
        except Exception:
            return False

    def start_gpu_profiling(self) -> None:
        """Start GPU profiling."""
        torch.cuda.profiler.start()

    def stop_gpu_profiling(self) -> None:
        """Stop GPU profiling."""
        torch.cuda.profiler.stop()

    def report_node_ip_and_gpu_id(self) -> tuple[str, int]:
        """Report the node IP and GPU ID of the current worker."""
        ip = ray._private.services.get_node_ip_address()
        # Workers that manage their own LOCAL_RANK will have an empty `ray.get_gpu_ids()`.
        gpu_ids = ray.get_gpu_ids()
        gpu_id = gpu_ids[0] if gpu_ids else torch.cuda.current_device()
        return (ip, gpu_id)

    # Temporary fix, 'data' is a kwarg due to some sort of ray bug
    @wrap_with_nvtx_name("policy_worker/get_reference_policy_logprobs")
    def get_reference_policy_logprobs(
        self,
        *,
        data: BatchedDataDict[Any],
        micro_batch_size: Optional[int] = None,
    ) -> BatchedDataDict[ReferenceLogprobOutputSpec]:
        """Get the logprobs from the reference policy for a batch of data.

        If micro_batch_size is provided, it will be used instead of the configured
        logprob_batch_size.

        Returns:
          a BatchedDataDict with key "reference_logprobs" and shape [batch_size, sequence_length].
          We use the convention that the logprob of the first token is 0 so that the sequence length is maintained.
          The logprob of input token i is specified at position i in the output logprobs tensor.
        """
        with self.use_reference_model():
            reference_logprobs = self.get_logprobs(
                data=data, micro_batch_size=micro_batch_size
            )

        return_data = BatchedDataDict[ReferenceLogprobOutputSpec]()
        return_data["reference_logprobs"] = reference_logprobs["logprobs"].cpu()
        return return_data

    def finish_training(self, *args: Any, **kwargs: Any) -> None:
        # Placeholder implementation
        pass
