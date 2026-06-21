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
import gc
import io
import os
import re
import time
import traceback
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, cast

import torch
import zmq

from nemo_rl.models.policy.utils import (
    IPCProtocol,
    calculate_aligned_size,
    rebuild_cuda_tensor_from_ipc,
)
from nemo_rl.utils import weight_transfer_sparse_codec as sparse_codec
from nemo_rl.utils.nsys import wrap_with_nvtx_name
from nemo_rl.utils.weight_transfer import packed_weight_transfer_consumer
from nemo_rl.utils.weight_transfer_protocol import (
    G_DENSE_TRANSPORT,
    G_SPARSE_INDICES_TRANSPORT,
    additive_weight_load_context,
    dtype_from_name,
)

try:
    import vllm  # noqa: F401
except ImportError:
    raise ImportError(
        "vLLM is not installed. Please check that the py_executable in the runtime_env of VllmGenerationWorker "
        "covers the vllm dependency. You may have to update nemo_rl/distributed/ray_actor_environment_registry.py. "
        "This error can also happen if the venv creation was aborted or errored out in the middle. In that case, "
        "please run at least once with the environment variable NRL_FORCE_REBUILD_VENVS=true set to force the rebuild of the environment."
    )


def fix_gpt_oss_export_transpose(key: str, weight: torch.Tensor) -> torch.Tensor:
    """Apply GPT-OSS down_proj transpose fix to the weight.

    This is a workaround for the issue that the down_proj layout is not the same across different frameworks.
        - HF needs [in, out] layout.
        - Megatron needs [in, out] layout.
        - vLLM needs [out, in] layout.
    See https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/3271 for more details.
    """
    if key.endswith("mlp.experts.down_proj"):
        weight = weight.transpose(-2, -1).contiguous()
    return weight


def fix_gemma3_vision_weight_name(key: str) -> str:
    """Re-insert the `vision_model` segment into Gemma3 vision-tower weights.

    When performing refit, the vision-tower weight paths are flattened. This unflattens them.
    """
    return re.sub(
        r"vision_tower\.(?!vision_model\.)", "vision_tower.vision_model.", key
    )


@dataclass(frozen=True)
class _SparseDeltaTargetPlan:
    target: torch.Tensor | None
    source_shape: tuple[int, ...] = ()
    source_strides: tuple[int, ...] = ()
    target_strides: tuple[int, ...] = ()
    source_to_target_dims: tuple[int, ...] | None = None
    fixed_target_indices: tuple[tuple[int, int], ...] = ()
    target_dim_offsets: tuple[tuple[int, int], ...] = ()
    shard_dim: int | None = None
    shard_start: int = 0
    shard_size: int = 0
    identity: bool = False
    skip: bool = False


class VllmInternalWorkerExtension:
    state_dict_info: dict[str, Any] | None = None
    delta_load_batch_size_bytes: int | None = None
    _direct_sparse_delta_targets: dict[str, torch.Tensor] | None = None
    _direct_sparse_delta_modules: dict[str, torch.nn.Module] | None = None
    _direct_sparse_delta_plan_cache: dict[str, _SparseDeltaTargetPlan | None] | None = (
        None
    )
    _direct_sparse_delta_load_enabled: bool = True
    _pending_http_dense_refit: bool = False

    def init_collective(
        self,
        rank_prefix: int,
        ip: str,
        port: int,
        world_size: int,
        train_world_size: int,
        local_world_size: int,
    ) -> None:
        """Initialize the collective communication."""
        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup

        if local_world_size <= 0:
            raise RuntimeError(
                f"Invalid vLLM local_world_size={local_world_size} for refit collective."
            )

        self_rank = getattr(self, "rank", None)
        torch_rank = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch_rank = int(torch.distributed.get_rank())

        raw_local_rank = int(self_rank if self_rank is not None else torch_rank or 0)
        local_rank = raw_local_rank % local_world_size
        # Place vLLM ranks after all training ranks so all training workers can join
        rank = train_world_size + rank_prefix + local_rank

        if rank < 0 or rank >= world_size:
            raise RuntimeError(
                "Invalid vLLM refit collective rank: "
                f"rank={rank}, world_size={world_size}, train_world_size={train_world_size}, "
                f"rank_prefix={rank_prefix}, raw_local_rank={raw_local_rank}, "
                f"local_rank={local_rank}, local_world_size={local_world_size}, "
                f"self_rank={self_rank}, torch_rank={torch_rank}"
            )

        if os.getenv("NRL_REFIT_DEBUG_RANKS") and (
            raw_local_rank != local_rank or self_rank != torch_rank
        ):
            print(
                "[refit_collective] "
                f"rank={rank}/{world_size}, train_world_size={train_world_size}, "
                f"rank_prefix={rank_prefix}, local_rank={local_rank}, "
                f"raw_local_rank={raw_local_rank}, local_world_size={local_world_size}, "
                f"self_rank={self_rank}, torch_rank={torch_rank}, device={self.device}",
                flush=True,
            )

        self.model_update_group = StatelessProcessGroup(  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored
            master_address=ip, port=port, rank=rank, world_size=world_size
        )
        self.model_update_group.init_nccl_communicator(device=self.device)

    def report_device_id(self) -> str:
        """Retrieve the UUID of the current CUDA device."""
        from nemo_rl.utils.nvml import get_device_uuid

        return get_device_uuid(self.device.index)

    def get_zmq_address(self):
        """Get the ZMQ address for the current device."""
        return f"ipc:///tmp/{self.report_device_id()}.sock"

    def maybe_init_zmq(self):
        """Initialize the ZMQ socket if it doesn't exist."""
        if not hasattr(self, "zmq_socket"):
            self.zmq_context = zmq.Context()  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored
            self.zmq_socket = self.zmq_context.socket(  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored
                zmq.REP
            )
            self.zmq_socket.setsockopt(
                zmq.SNDTIMEO, 120000
            )  # set timeout to 120 seconds
            self.zmq_socket.setsockopt(
                zmq.RCVTIMEO, 120000
            )  # set timeout to 120 seconds
            self.zmq_socket.setsockopt(zmq.LINGER, 0)
            self.zmq_socket.connect(self.get_zmq_address())

    def prepare_refit_info(
        self,
        state_dict_info: dict[str, Any],
        delta_load_batch_size_bytes: int | None = None,
        direct_sparse_delta_load: bool = True,
    ) -> None:
        """Prepare state dict metadata for IPC/ZMQ weight refitting.

        Collective refit receives tensor metadata from the transfer headers.

        Args:
            state_dict_info (dict): A dictionary containing the info for refit.
                e.g. {tensor_name: (shape, dtype)}
            delta_load_batch_size_bytes (int | None): Maximum decoded delta bytes
                to batch before calling vLLM load_weights. None means delta
                transfer is disabled.
            direct_sparse_delta_load (bool): Whether sparse deltas may be
                applied directly to compatible vLLM tensors.
        """
        self.state_dict_info = state_dict_info
        self.delta_load_batch_size_bytes = delta_load_batch_size_bytes
        self._direct_sparse_delta_load_enabled = bool(direct_sparse_delta_load)
        self._direct_sparse_delta_targets = None
        self._direct_sparse_delta_modules = None
        self._direct_sparse_delta_plan_cache = None

    def _process_weights_after_loading(
        self,
        model_config: Any,
        target_device: torch.device,
    ) -> None:
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.utils import (
            process_weights_after_loading,
        )

        with set_current_vllm_config(self.model_runner.vllm_config):
            process_weights_after_loading(
                self.model_runner.model,
                model_config,
                target_device,
            )

    @staticmethod
    def _split_policy_and_draft_weights(
        weights: list[tuple[str, torch.Tensor]],
    ) -> tuple[list[tuple[str, torch.Tensor]], list[tuple[str, torch.Tensor]]]:
        """Split trainer-owned draft weights from policy weights.

        This path is only used for the Eagle3 online-training flow, where the
        trainer exports draft parameters under a `draft.` prefix before sending
        them to vLLM.
        This implementation is specific to the eagle model. For MTP, we can add
        similar logic to this function to split weights and send it to the drafter.
        The "draft." prefix is added here https://github.com/isomap/RL/blob/d3a5e1396d00f82fb888d9ec6800687a23bb4017/nemo_rl/models/policy/workers/megatron_policy_worker.py#L967-L997
        """
        policy_weights = []
        draft_weights = []
        for key, tensor in weights:
            if key.startswith("draft."):
                draft_weights.append((key.removeprefix("draft."), tensor))
            else:
                policy_weights.append((key, tensor))
        return policy_weights, draft_weights

    @staticmethod
    def _trim_vocab_padding(
        draft_model: torch.nn.Module,
        draft_weights: list[tuple[str, torch.Tensor]],
    ) -> list[tuple[str, torch.Tensor]]:
        """Trim padded vocab dimensions from draft weights.

        Megatron pads vocab to a multiple, but vLLM 0.20's autoloader
        strictly asserts loaded_weight.shape[0] == org_vocab_size on
        VocabParallelEmbedding layers. Each such layer may have a
        different org_vocab_size (e.g. embed_tokens uses vocab_size
        while lm_head uses draft_vocab_size), so we match each weight
        to its target module by name.
        """
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding,
        )

        vocab_sizes: dict[str, int] = {}
        for name, module in draft_model.named_modules():
            if isinstance(module, VocabParallelEmbedding):
                vocab_sizes[name] = module.org_vocab_size

        if not vocab_sizes:
            return draft_weights

        trimmed = []
        for key, tensor in draft_weights:
            for mod_name, org_vocab_size in vocab_sizes.items():
                leaf = mod_name.rsplit(".", 1)[-1]
                if leaf in key and tensor.shape[0] > org_vocab_size:
                    tensor = tensor[:org_vocab_size]
                    break
            trimmed.append((key, tensor))
        return trimmed

    def _load_draft_weights(
        self, draft_weights: list[tuple[str, torch.Tensor]]
    ) -> None:
        if not draft_weights:
            return

        draft_owner = getattr(self.model_runner, "drafter", None)
        draft_model = getattr(draft_owner, "model", None) if draft_owner else None

        if draft_model is None:
            print(
                "[draft] Received draft weights but vLLM drafter is unavailable; skipping draft update."
            )
            return
        draft_weights = self._trim_vocab_padding(draft_model, draft_weights)
        draft_model.load_weights(weights=draft_weights)

    def _load_weights(self, weights: list[tuple[str, torch.Tensor]]) -> None:
        """Load weights with GPT-OSS, Gemma3, FP8, and draft-weight support.

        Applies GPT-OSS down_proj transpose and Gemma3 vision-tower weight name
        fixes if needed, splits policy/draft weights, applies FP8 conversion if
        needed, and loads draft weights into the drafter model.
        """
        from nemo_rl.models.generation.vllm.quantization import fp8

        if (
            "GptOssForCausalLM"
            in self.model_runner.vllm_config.model_config.architectures
        ):
            for idx, (key, weight) in enumerate(weights):
                weight = fix_gpt_oss_export_transpose(key, weight)
                weights[idx] = (key, weight)

        if (
            "Gemma3ForConditionalGeneration"
            in self.model_runner.vllm_config.model_config.architectures
        ):
            for idx, (key, weight) in enumerate(weights):
                weights[idx] = (fix_gemma3_vision_weight_name(key), weight)

        policy_weights, draft_weights = self._split_policy_and_draft_weights(weights)
        if fp8.is_fp8_model(self.model_runner.vllm_config):
            fp8.load_weights(policy_weights, self.model_runner)
        else:
            self.model_runner.model.load_weights(weights=policy_weights)

        self._load_draft_weights(draft_weights)

    def _iter_delta_load_target_tensors(self) -> Iterator[torch.Tensor]:
        yield from self.model_runner.model.parameters()
        yield from self.model_runner.model.buffers()

    def _load_weight_deltas(self, weights: list[tuple[str, torch.Tensor]]) -> None:
        """Apply additive weight deltas through vLLM's regular loaders."""
        with additive_weight_load_context(self._iter_delta_load_target_tensors()):
            self._load_weights(weights)

    def _load_sparse_weight_deltas(
        self,
        payload_tensors: list[tuple[str, torch.Tensor]],
        metadata: list[dict[str, Any]],
    ) -> bool | list[dict[str, Any]]:
        """Apply sparse deltas directly to compatible vLLM tensors."""
        if not self._direct_sparse_delta_load_enabled:
            return list(metadata)
        if self._direct_sparse_delta_uses_loader_transform():
            return list(metadata)

        targets = self._direct_sparse_delta_target_map()
        raw_locations, raw_values = sparse_codec.packed_sparse_payload_tensors(
            payload_tensors
        )
        missing: list[dict[str, Any]] = []
        spans = sparse_codec.sparse_metadata_spans(metadata)
        with torch.no_grad():
            for item, span in zip(metadata, spans, strict=True):
                plan = self._cached_direct_sparse_delta_target_plan(item, targets)
                if plan is None:
                    missing.append(item)
                    continue
                if plan.skip:
                    continue
                target = plan.target
                assert target is not None
                index_start, index_end, value_start, value_end = span
                values = raw_values[value_start:value_end].to(
                    device=target.device,
                    dtype=target.dtype,
                    non_blocking=True,
                )
                sparse_range = sparse_codec.sparse_range_for_item(item, span)
                if plan.identity and sparse_range is not None:
                    range_start, range_count = sparse_range
                    if range_count == 0:
                        continue
                    target_slice = target.data.view(-1).narrow(
                        0, range_start, range_count
                    )
                    target_slice.add_(values)
                    continue
                locations = sparse_codec.sparse_locations_for_item(
                    item,
                    raw_locations,
                    (index_start, index_end, value_start, value_end),
                    device=target.device,
                    dtype=torch.long,
                )
                value_count = value_end - value_start
                if int(locations.numel()) != value_count:
                    raise RuntimeError(
                        "Sparse direct delta location/value mismatch for "
                        f"{item['name']!r}: "
                        f"encoding={sparse_codec.sparse_index_encoding_for_item(item)!r} "
                        f"index_span={index_end - index_start} "
                        f"value_span={value_count}"
                    )
                locations, values = self._local_sparse_delta_update_inputs(
                    locations,
                    values,
                    plan,
                )
                if locations.numel() == 0:
                    continue
                target_flat = target.data.view(-1)
                target_flat.index_add_(0, locations, values)

        if missing:
            return missing
        return True

    def _load_sparse_weight_deltas_from_transfer(
        self,
        payload_tensors: list[tuple[str, torch.Tensor]],
        metadata: list[dict[str, Any]],
    ) -> None:
        sparse_load_result = self._load_sparse_weight_deltas(payload_tensors, metadata)
        if sparse_load_result is True:
            return

        byte_cap = self.delta_load_batch_size_bytes
        if byte_cap is None:
            byte_cap = 512 * 1024**2
        self._apply_sparse_delta_fallback(
            payload_tensors, metadata, sparse_load_result, byte_cap
        )

    def _apply_sparse_delta_fallback(
        self,
        payload_tensors: list[tuple[str, torch.Tensor]],
        metadata: list[dict[str, Any]],
        sparse_load_result: bool | list[dict[str, Any]],
        byte_cap: int,
    ) -> tuple[int, int]:
        """Decode and additively load the deltas the direct path could not place.

        Returns ``(fallback_tensor_count, fallback_batch_count)``.
        """
        fallback_metadata = (
            sparse_load_result if isinstance(sparse_load_result, list) else metadata
        )
        batches = 0
        for batch in sparse_codec.decode_sparse(
            payload_tensors,
            fallback_metadata,
            self.device,
            byte_cap,
        ):
            self._load_weight_deltas(batch)
            batches += 1
        return len(fallback_metadata), batches

    def _direct_sparse_delta_target_map(self) -> dict[str, torch.Tensor]:
        if self._direct_sparse_delta_targets is None:
            self._direct_sparse_delta_targets = dict(
                self.model_runner.model.named_parameters()
            )
            self._direct_sparse_delta_targets.update(
                self.model_runner.model.named_buffers()
            )
        return self._direct_sparse_delta_targets

    def _direct_sparse_delta_module(
        self,
        parameter_name: str,
    ) -> torch.nn.Module | None:
        if self._direct_sparse_delta_modules is None:
            self._direct_sparse_delta_modules = dict(
                self.model_runner.model.named_modules()
            )
        module_name = parameter_name.rsplit(".", 1)[0]
        return self._direct_sparse_delta_modules.get(module_name)

    def _direct_sparse_delta_uses_loader_transform(self) -> bool:
        architectures = self.model_runner.vllm_config.model_config.architectures
        if any(
            architecture in architectures
            for architecture in ("GptOssForCausalLM", "Gemma3ForConditionalGeneration")
        ):
            return True

        from nemo_rl.models.generation.vllm.quantization import fp8

        return fp8.is_fp8_model(self.model_runner.vllm_config)

    def _cached_direct_sparse_delta_target_plan(
        self,
        item: dict[str, Any],
        targets: dict[str, torch.Tensor],
    ) -> _SparseDeltaTargetPlan | None:
        name = str(item["name"])
        if self._direct_sparse_delta_plan_cache is None:
            self._direct_sparse_delta_plan_cache = {}
        if name not in self._direct_sparse_delta_plan_cache:
            self._direct_sparse_delta_plan_cache[name] = (
                self._direct_sparse_delta_target_plan(item, targets)
            )
        return self._direct_sparse_delta_plan_cache[name]

    def _direct_sparse_delta_target_plan(
        self,
        item: dict[str, Any],
        targets: dict[str, torch.Tensor],
    ) -> _SparseDeltaTargetPlan | None:
        name = str(item["name"])
        if name.startswith("mtp."):
            return _SparseDeltaTargetPlan(target=None, skip=True)
        target_name = self._map_direct_sparse_delta_name(name)
        if target_name is None:
            return None
        if any(f".{candidate}_proj." in target_name for candidate in ("q", "k", "v")):
            return self._direct_sparse_delta_qkv_plan(item, target_name, targets)
        if ".experts." in target_name:
            expert_plan = self._direct_sparse_delta_expert_plan(
                item, target_name, targets
            )
            if expert_plan is not None or re.match(
                r"^.*\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight$",
                target_name,
            ):
                return expert_plan

        target = targets.get(target_name)
        if target is None:
            return None
        if not self._direct_sparse_delta_target_is_compatible(target, item):
            return None

        source_shape = self._direct_sparse_delta_shape_tuple(item)
        target_shape = tuple(int(dim) for dim in target.shape)
        if target_shape == source_shape:
            if target.numel() != int(item["numel"]):
                return None
            return self._make_sparse_delta_target_plan(target, source_shape)

        shard_plan = self._direct_sparse_delta_shard_plan(item, target)
        if shard_plan is not None:
            return shard_plan
        return None

    def _direct_sparse_delta_qkv_plan(
        self,
        item: dict[str, Any],
        target_name: str,
        targets: dict[str, torch.Tensor],
    ) -> _SparseDeltaTargetPlan | None:
        shard_id = None
        packed_name = target_name
        for candidate in ("q", "k", "v"):
            needle = f".{candidate}_proj."
            if needle in target_name:
                shard_id = candidate
                packed_name = target_name.replace(needle, ".qkv_proj.", 1)
                break
        if shard_id is None:
            return None

        target = targets.get(packed_name)
        if target is None:
            return None
        if not self._direct_sparse_delta_target_is_compatible(target, item):
            return None
        module = self._direct_sparse_delta_module(packed_name)
        if module is None:
            return None

        output_dim = getattr(target, "output_dim", None)
        if not isinstance(output_dim, int):
            return None
        if output_dim < 0:
            output_dim += target.ndim
        if output_dim < 0 or output_dim >= target.ndim:
            return None

        required_attrs = (
            "num_heads",
            "num_kv_heads",
            "head_size",
            "v_head_size",
            "tp_rank",
            "num_kv_head_replicas",
        )
        if not all(hasattr(module, attr) for attr in required_attrs):
            return None

        module_attrs = cast(Any, module)
        num_heads = int(module_attrs.num_heads)
        num_kv_heads = int(module_attrs.num_kv_heads)
        head_size = int(module_attrs.head_size)
        v_head_size = int(module_attrs.v_head_size)
        tp_rank = int(module_attrs.tp_rank)
        kv_replicas = int(module_attrs.num_kv_head_replicas)

        if shard_id == "q":
            shard_offset = 0
            shard_size = num_heads * head_size
            shard_rank = tp_rank
        elif shard_id == "k":
            shard_offset = num_heads * head_size
            shard_size = num_kv_heads * head_size
            shard_rank = tp_rank // kv_replicas
        else:
            shard_offset = (num_heads + num_kv_heads) * head_size
            shard_size = num_kv_heads * v_head_size
            shard_rank = tp_rank // kv_replicas

        source_shape = self._direct_sparse_delta_shape_tuple(item)
        target_shape = tuple(int(dim) for dim in target.shape)
        if len(source_shape) != len(target_shape):
            return None
        if any(
            source_dim != target_dim
            for dim, (source_dim, target_dim) in enumerate(
                zip(source_shape, target_shape, strict=True)
            )
            if dim != output_dim
        ):
            return None
        if target_shape[output_dim] < shard_offset + shard_size:
            return None

        shard_start = shard_rank * shard_size
        if source_shape[output_dim] < shard_start:
            return _SparseDeltaTargetPlan(target=None, skip=True)
        return self._make_sparse_delta_target_plan(
            target,
            source_shape=source_shape,
            shard_dim=output_dim,
            shard_start=shard_start,
            shard_size=min(shard_size, source_shape[output_dim] - shard_start),
            target_dim_offsets=((output_dim, shard_offset),),
        )

    def _direct_sparse_delta_expert_plan(
        self,
        item: dict[str, Any],
        target_name: str,
        targets: dict[str, torch.Tensor],
    ) -> _SparseDeltaTargetPlan | None:
        match = re.match(
            r"^(?P<prefix>.*\.experts)\.(?P<expert>\d+)\."
            r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<leaf>weight)$",
            target_name,
        )
        if match is None:
            return None

        prefix = match.group("prefix")
        global_expert_id = int(match.group("expert"))
        proj = match.group("proj")
        leaf = match.group("leaf")
        if proj == "gate_proj":
            packed_name = f"{prefix}.w13_{leaf}"
            shard_id = "w1"
        elif proj == "up_proj":
            packed_name = f"{prefix}.w13_{leaf}"
            shard_id = "w3"
        else:
            packed_name = f"{prefix}.w2_{leaf}"
            shard_id = "w2"

        target = targets.get(packed_name)
        if target is None:
            return None
        if not self._direct_sparse_delta_target_is_compatible(target, item):
            return None
        module = self._direct_sparse_delta_module(packed_name)
        if module is None:
            return None

        local_expert_id = global_expert_id
        map_expert = getattr(module, "_map_global_expert_id_to_local_expert_id", None)
        if callable(map_expert):
            local_expert_id = int(cast(Any, map_expert)(global_expert_id))
        if local_expert_id < 0:
            return _SparseDeltaTargetPlan(target=None, skip=True)

        source_shape = self._direct_sparse_delta_shape_tuple(item)
        target_shape = tuple(int(dim) for dim in target.shape)
        if len(target_shape) != len(source_shape) + 1:
            return None
        if local_expert_id >= target_shape[0]:
            return None

        shard_dim = 1 if shard_id == "w2" else 0
        target_shard_dim = shard_dim + 1
        if any(
            source_dim != target_shape[dim + 1]
            for dim, source_dim in enumerate(source_shape)
            if dim != shard_dim
        ):
            return None

        shard_size = target_shape[target_shard_dim]
        moe_config = getattr(module, "moe_config", None)
        if shard_id in ("w1", "w3") and bool(
            getattr(moe_config, "is_act_and_mul", False)
        ):
            shard_size //= 2
        target_dim_offsets = (
            ((target_shard_dim, shard_size),) if shard_id == "w3" else ()
        )
        if any(
            target_shape[offset_dim] < offset + shard_size
            for offset_dim, offset in target_dim_offsets
        ):
            return None
        tp_rank = int(getattr(module, "tp_rank", self._direct_sparse_delta_tp_rank(1)))
        shard_start = tp_rank * shard_size
        if source_shape[shard_dim] < shard_start:
            return _SparseDeltaTargetPlan(target=None, skip=True)

        return self._make_sparse_delta_target_plan(
            target,
            source_shape=source_shape,
            source_to_target_dims=tuple(dim + 1 for dim in range(len(source_shape))),
            fixed_target_indices=((0, local_expert_id),),
            target_dim_offsets=target_dim_offsets,
            shard_dim=shard_dim,
            shard_start=shard_start,
            shard_size=min(shard_size, source_shape[shard_dim] - shard_start),
        )

    @staticmethod
    def _direct_sparse_delta_target_is_compatible(
        target: torch.Tensor,
        item: dict[str, Any],
    ) -> bool:
        expected_dtype = dtype_from_name(str(item["dtype"]))
        return (
            target.is_contiguous()
            and target.dtype.is_floating_point
            and expected_dtype.is_floating_point
        )

    def _make_sparse_delta_target_plan(
        self,
        target: torch.Tensor,
        source_shape: tuple[int, ...],
        *,
        source_to_target_dims: tuple[int, ...] | None = None,
        fixed_target_indices: tuple[tuple[int, int], ...] = (),
        target_dim_offsets: tuple[tuple[int, int], ...] = (),
        shard_dim: int | None = None,
        shard_start: int = 0,
        shard_size: int = 0,
    ) -> _SparseDeltaTargetPlan:
        if source_to_target_dims is None:
            source_to_target_dims = tuple(range(len(source_shape)))
        target_shape = tuple(int(dim) for dim in target.shape)
        identity = (
            shard_dim is None
            and not fixed_target_indices
            and not target_dim_offsets
            and source_to_target_dims == tuple(range(len(source_shape)))
            and source_shape == target_shape
        )
        return _SparseDeltaTargetPlan(
            target=target,
            source_shape=source_shape,
            source_strides=tuple(self._contiguous_strides(source_shape)),
            target_strides=tuple(self._contiguous_strides(target_shape)),
            source_to_target_dims=source_to_target_dims,
            fixed_target_indices=fixed_target_indices,
            target_dim_offsets=target_dim_offsets,
            shard_dim=shard_dim,
            shard_start=shard_start,
            shard_size=shard_size,
            identity=identity,
        )

    def _direct_sparse_delta_shard_plan(
        self,
        item: dict[str, Any],
        target: torch.Tensor,
    ) -> _SparseDeltaTargetPlan | None:
        source_shape = self._direct_sparse_delta_shape_tuple(item)
        target_shape = tuple(int(dim) for dim in target.shape)
        if len(source_shape) != len(target_shape):
            return None

        candidate_dims = []
        for attr_name in ("output_dim", "input_dim"):
            shard_dim = getattr(target, attr_name, None)
            if isinstance(shard_dim, int):
                if shard_dim < 0:
                    shard_dim += len(source_shape)
            if isinstance(shard_dim, int) and shard_dim not in candidate_dims:
                candidate_dims.append(shard_dim)
        if not candidate_dims:
            differing_dims = [
                dim
                for dim, (source_dim, target_dim) in enumerate(
                    zip(source_shape, target_shape, strict=True)
                )
                if source_dim != target_dim
            ]
            if len(differing_dims) == 1:
                candidate_dims = differing_dims

        base_tp_size = int(getattr(target, "tp_size", 1))
        base_tp_rank = int(
            getattr(
                target,
                "tp_rank",
                self._direct_sparse_delta_tp_rank(base_tp_size),
            )
        )

        for shard_dim in candidate_dims:
            if shard_dim < 0 or shard_dim >= len(source_shape):
                continue
            tp_size = base_tp_size
            tp_rank = base_tp_rank
            if tp_size <= 1:
                target_dim = target_shape[shard_dim]
                source_dim = source_shape[shard_dim]
                if target_dim <= 0 or source_dim % target_dim != 0:
                    continue
                tp_size = source_dim // target_dim
                if tp_size <= 1:
                    continue
                tp_rank = self._direct_sparse_delta_tp_rank(tp_size)
            if any(
                source_dim != target_dim
                for dim, (source_dim, target_dim) in enumerate(
                    zip(source_shape, target_shape, strict=True)
                )
                if dim != shard_dim
            ):
                continue
            shard_size = target_shape[shard_dim]
            if source_shape[shard_dim] > shard_size * tp_size:
                continue
            return self._make_sparse_delta_target_plan(
                target=target,
                source_shape=source_shape,
                shard_dim=shard_dim,
                shard_start=tp_rank * shard_size,
                shard_size=shard_size,
            )
        return None

    @staticmethod
    def _contiguous_strides(shape: tuple[int, ...]) -> list[int]:
        stride = 1
        strides = [1 for _ in shape]
        for dim in range(len(shape) - 1, -1, -1):
            strides[dim] = stride
            stride *= shape[dim]
        return strides

    def _local_sparse_delta_update_inputs(
        self,
        locations: torch.Tensor,
        values: torch.Tensor,
        plan: _SparseDeltaTargetPlan,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if plan.identity:
            return locations, values

        source_shape = plan.source_shape
        source_strides = plan.source_strides
        target_strides = plan.target_strides
        selected_locations = locations
        selected_values = values

        if plan.shard_dim is not None:
            shard_dim = plan.shard_dim
            shard_coords = torch.div(
                locations,
                source_strides[shard_dim],
                rounding_mode="floor",
            ).remainder(source_shape[shard_dim])
            shard_end = min(
                plan.shard_start + plan.shard_size,
                source_shape[shard_dim],
            )
            keep = (shard_coords >= plan.shard_start) & (shard_coords < shard_end)
            selected_locations = locations[keep]
            selected_values = values[keep]
            if selected_locations.numel() == 0:
                return selected_locations, selected_values

        base_offset = 0
        for target_dim, index in plan.fixed_target_indices:
            base_offset += index * target_strides[target_dim]

        local_locations: torch.Tensor | None = None
        source_to_target_dims = plan.source_to_target_dims
        assert source_to_target_dims is not None
        for dim, source_stride in enumerate(source_strides):
            coord = torch.div(
                selected_locations,
                source_stride,
                rounding_mode="floor",
            ).remainder(source_shape[dim])
            if dim == plan.shard_dim:
                coord = coord - plan.shard_start
            target_dim = source_to_target_dims[dim]
            for offset_dim, offset in plan.target_dim_offsets:
                if offset_dim == target_dim:
                    coord = coord + offset
                    break
            contribution = coord * target_strides[target_dim]
            if local_locations is None:
                local_locations = contribution
                if base_offset:
                    local_locations += base_offset
            else:
                local_locations += contribution

        if local_locations is None:
            local_locations = torch.empty_like(selected_locations)
            local_locations.fill_(base_offset)
        return local_locations, selected_values

    def _direct_sparse_delta_tp_rank(self, tp_size: int) -> int:
        if tp_size <= 1:
            return 0
        rank = int(getattr(self, "rank", 0))
        return rank % tp_size

    @staticmethod
    def _direct_sparse_delta_shape_tuple(item: dict[str, Any]) -> tuple[int, ...]:
        return tuple(int(dim) for dim in item["shape"])

    def _map_direct_sparse_delta_name(self, name: str) -> str | None:
        mapper = getattr(self.model_runner.model, "hf_to_vllm_mapper", None)
        if mapper is not None:
            map_name = getattr(mapper, "_map_name", None)
            if callable(map_name):
                return cast(str, map_name(name))

            for old, new in getattr(mapper, "orig_to_new_substr", {}).items():
                if old in name:
                    if new is None:
                        return None
                    name = name.replace(old, new, 1)
            for old, new in getattr(mapper, "orig_to_new_prefix", {}).items():
                if name.startswith(old):
                    if new is None:
                        return None
                    name = name.replace(old, new, 1)
            for old, new in getattr(mapper, "orig_to_new_suffix", {}).items():
                if name.endswith(old):
                    if new is None:
                        return None
                    name = new.join(name.rsplit(old, 1))
        if name.startswith("draft."):
            return None
        return name

    @wrap_with_nvtx_name("vllm_internal_worker_extension/update_weights_via_ipc_zmq")
    def update_weights_via_ipc_zmq(self) -> bool:
        """Receive and update model weights via ZMQ IPC socket.

        Returns:
            bool: True if weights were successfully updated.
        """
        state_dict_info = self.state_dict_info
        assert state_dict_info is not None, (
            "state_dict_info is not prepared. "
            "Please call prepare_refit_info when initializing the worker."
        )

        try:
            self.maybe_init_zmq()
            while True:
                # Blocking receive with timeout (this is the main operation).
                payload = self.zmq_socket.recv_pyobj()

                if payload == IPCProtocol.COMPLETE:
                    # COMPLETE means the update is done; run vLLM post-load hooks before ACK.
                    self._process_weights_after_loading(self.model_config, self.device)
                    self.zmq_socket.send(IPCProtocol.ACK.value.encode())
                    break

                ipc_handle, list_keys, used_bytes = payload
                buffer = rebuild_cuda_tensor_from_ipc(ipc_handle, self.device.index)

                weights = []
                offset = 0
                for key in list_keys:
                    shape, dtype = state_dict_info[key]
                    if isinstance(shape, list):
                        shape = torch.Size(shape)

                    size_in_bytes = dtype.itemsize * shape.numel()
                    # Get the weight from the buffer.
                    weights.append(
                        (
                            key,
                            buffer[offset : offset + size_in_bytes]
                            .view(dtype=dtype)
                            .view(shape),
                        )
                    )

                    aligned_size = calculate_aligned_size(size_in_bytes)
                    # Move offset to the next weight.
                    offset += aligned_size

                assert offset == used_bytes, (
                    "Offset is not equal to used bytes, usually indicate inaccurate info like keys or cached dtype in state_dict_info"
                )

                # Load weights into the model.
                self._load_weights(weights)

                torch.cuda.current_stream().synchronize()

                # CRITICAL: Delete views before ACK to prevent corruption.
                # 'weights' contains views into IPC shared memory. Even though load_weights()
                # copied the data, Python may not garbage collect these view objects immediately.
                # If sender reuses the buffer before GC runs, old views would read corrupted data.
                # Explicit del ensures immediate cleanup before sending ACK.
                del weights, buffer
                self.zmq_socket.send(IPCProtocol.ACK.value.encode())

            gc.collect()
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            print(
                f"Error in VllmInternalWorkerExtension.update_weights_via_ipc_zmq: {e}.\n"
                f"{traceback.format_exc()}"
            )
            return False

    @wrap_with_nvtx_name(
        "vllm_internal_worker_extension/update_weights_from_collective"
    )
    def update_weights_from_collective(self) -> bool:
        """Update the model weights from collective communication."""
        try:
            state_dict_info = self.state_dict_info
            assert state_dict_info is not None, (
                "state_dict_info is not prepared. "
                "Please call prepare_refit_info when initializing the worker."
            )
            transfer_result = packed_weight_transfer_consumer(
                group=self.model_update_group,
                src=0,
                load_full_weights_func=self._load_weights,
                load_sparse_weights_func=self._load_sparse_weight_deltas_from_transfer,
                device=self.device,
            )
            if transfer_result.loaded_any and not transfer_result.is_delta_sync:
                # Process weights after full loads for FP8 KV cache and vLLM post-load hooks.
                self._process_weights_after_loading(
                    self.model_runner.model_config,
                    next(self.model_runner.model.parameters()).device,
                )

        except Exception as e:
            print(
                f"Error in VllmInternalWorkerExtension.update_weights_from_collective: {e}\n"
                f"{traceback.format_exc()}"
            )
            return False

        return True

    @wrap_with_nvtx_name(
        "vllm_internal_worker_extension/update_weights_from_serialized_sparse_payload"
    )
    def update_weights_from_serialized_sparse_payload(
        self,
        serialized_payload: bytes,
        synchronize: bool = True,
    ) -> dict[str, Any]:
        """Apply one serialized sparse-delta payload received over HTTP.

        The payload format mirrors the sparse payload consumed by collective
        refit, but is transported as a per-bucket torch-serialized object so a
        remote HTTP caller can stream buckets across Kubernetes clusters.
        """
        try:
            started = time.perf_counter()
            request = self._load_serialized_sparse_payload(serialized_payload)
            deserialize_s = time.perf_counter() - started
            result = self._update_weights_from_sparse_request(
                request,
                synchronize=synchronize,
            )
            result["receiver_deserialize_s"] = deserialize_s
            result["receiver_total_s"] = time.perf_counter() - started
            return result
        except Exception as e:
            print(
                "Error in "
                "VllmInternalWorkerExtension.update_weights_from_serialized_sparse_payload: "
                f"{e}\n{traceback.format_exc()}"
            )
            return {"ok": False, "error": str(e)}

    @staticmethod
    def _load_serialized_sparse_payload(serialized_payload: bytes) -> dict[str, Any]:
        request = torch.load(
            io.BytesIO(serialized_payload),
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(request, dict):
            raise ValueError("Serialized sparse refit payload must decode to a dict.")
        return request

    def _update_weights_from_sparse_request(
        self,
        request: dict[str, Any],
        *,
        synchronize: bool,
    ) -> dict[str, Any]:
        transport = request.get("transport", G_SPARSE_INDICES_TRANSPORT)
        metadata = [dict(item) for item in request.get("metadata", [])]
        payload_tensors = request["payload_tensors"]
        if transport == G_DENSE_TRANSPORT:
            if metadata:
                raise ValueError("Dense HTTP refit payloads cannot carry metadata.")
            load_started = time.perf_counter()
            self._load_weights(payload_tensors)
            self._pending_http_dense_refit = True
            dense_load_s = time.perf_counter() - load_started
            sync_s = self._synchronize_refit_device(synchronize)
            return {
                "ok": True,
                "full_tensors": len(payload_tensors),
                "receiver_dense_load_s": dense_load_s,
                "receiver_sync_s": sync_s,
            }
        if transport != G_SPARSE_INDICES_TRANSPORT:
            raise ValueError(f"Unsupported HTTP refit transport: {transport!r}")

        direct_started = time.perf_counter()
        sparse_load_result = self._load_sparse_weight_deltas(
            payload_tensors,
            metadata,
        )
        direct_apply_s = time.perf_counter() - direct_started
        direct_tensors = len(metadata)
        fallback_tensors = 0
        fallback_batches = 0
        fallback_apply_s = 0.0
        if sparse_load_result is not True:
            byte_cap = self.delta_load_batch_size_bytes
            if byte_cap is None:
                byte_cap = int(
                    request.get("delta_load_batch_size_bytes", 512 * 1024**2)
                )
            fallback_started = time.perf_counter()
            fallback_tensors, fallback_batches = self._apply_sparse_delta_fallback(
                payload_tensors, metadata, sparse_load_result, byte_cap
            )
            direct_tensors = len(metadata) - fallback_tensors
            fallback_apply_s = time.perf_counter() - fallback_started

        sync_s = self._synchronize_refit_device(synchronize)
        result = {
            "ok": True,
            "direct_tensors": direct_tensors,
            "fallback_tensors": fallback_tensors,
            "fallback_batches": fallback_batches,
            "receiver_direct_apply_s": direct_apply_s,
            "receiver_fallback_apply_s": fallback_apply_s,
            "receiver_sync_s": sync_s,
        }
        return result

    def _synchronize_refit_device(self, synchronize: bool) -> float:
        if not synchronize or not torch.cuda.is_available():
            return 0.0
        sync_started = time.perf_counter()
        torch.cuda.synchronize(self.device)
        return time.perf_counter() - sync_started

    def synchronize_device(self) -> dict[str, Any]:
        """Synchronize this vLLM worker's CUDA device after deferred refit applies."""
        process_s = 0.0
        if self._pending_http_dense_refit:
            process_started = time.perf_counter()
            self._process_weights_after_loading(
                self.model_runner.model_config,
                next(self.model_runner.model.parameters()).device,
            )
            self._pending_http_dense_refit = False
            process_s = time.perf_counter() - process_started
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        return {"ok": True, "receiver_dense_process_s": process_s}

    def cleanup(self) -> None:
        """Shutdown and cleanup resources."""
        # Close ZMQ socket and context if they exist
        if hasattr(self, "zmq_socket"):
            self.zmq_socket.close()
            self.zmq_context.term()

    def start_gpu_profiling(self) -> None:
        """Start GPU profiling."""
        torch.cuda.profiler.start()

    def stop_gpu_profiling(self) -> None:
        """Stop GPU profiling."""
        torch.cuda.profiler.stop()
