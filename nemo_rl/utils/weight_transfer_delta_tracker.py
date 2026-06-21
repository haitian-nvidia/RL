# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from nemo_rl.utils import weight_transfer_sparse_codec as sparse_codec
from nemo_rl.utils.weight_transfer_protocol import (
    G_REFIT_BASELINE_IN_MEMORY_ENV,
    G_REFIT_BASELINE_MMAP_DIR_ENV,
    G_REFIT_PREWARM_DELTA_BASELINE_ENV,
    NamedTensor,
    TensorBatch,
    TensorMetadata,
    TensorPayload,
    config_env_flag,
    config_to_dict,
    dtype_from_name,
    dtype_itemsize,
)


@dataclass(slots=True)
class _BaselineEntry:
    tensor: torch.Tensor
    path: str | None


@dataclass(slots=True)
class _PendingSparseUpdate:
    locations: torch.Tensor | None
    values: torch.Tensor
    range_start: int | None = None


class DeltaCompressionTracker:
    """Source-side mmap baseline for sparse-delta vLLM refit."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        config = config_to_dict(config)
        self.full_sync_interval = int(config["full_sync_interval"])
        self.sparse_bucket_size_bytes = int(config["sparse_bucket_size_bytes"])
        if self.full_sync_interval < 1:
            raise ValueError("delta_compression.full_sync_interval must be >= 1")
        if self.sparse_bucket_size_bytes < 1:
            raise ValueError("delta_compression.sparse_bucket_size_bytes must be >= 1")
        self.index_encoding = sparse_codec.validate_sparse_index_encoding(
            str(config["index_encoding"])
            if "index_encoding" in config
            else sparse_codec.G_INDEX_ENCODING_INDICES
        )
        self.delta_dtype = dtype_from_name(str(config["dtype"]).lower())
        self.prewarm_baseline = config_env_flag(
            config,
            "prewarm_baseline",
            env_name=G_REFIT_PREWARM_DELTA_BASELINE_ENV,
            default=True,
        )
        self.baseline_in_memory = config_env_flag(
            config,
            "baseline_in_memory",
            env_name=G_REFIT_BASELINE_IN_MEMORY_ENV,
            default=False,
        )
        self.baseline_mmap_dir = config.get("baseline_mmap_dir") or os.getenv(
            G_REFIT_BASELINE_MMAP_DIR_ENV
        )
        self.async_receiver_apply = config_env_flag(
            config,
            "async_receiver_apply",
            env_name="NRL_REFIT_ASYNC_RECEIVER_APPLY",
            default=True,
        )
        self.committed_syncs = 0
        self.baseline: dict[str, torch.Tensor] = {}
        self._entries: dict[str, _BaselineEntry] = {}
        self._pending_full_sync_names: set[str] = set()
        self._pending_updates: dict[str, _PendingSparseUpdate] = {}

    def __del__(self) -> None:
        for entry in getattr(self, "_entries", {}).values():
            if entry.path:
                try:
                    os.unlink(entry.path)
                except FileNotFoundError:
                    pass

    def should_prewarm_baseline(self) -> bool:
        return self.full_sync_interval > 1 and self.prewarm_baseline

    def is_delta_sync(self) -> bool:
        return (
            self.full_sync_interval > 1
            and self.committed_syncs > 0
            and self.committed_syncs % self.full_sync_interval != 0
        )

    def has_pending_full_sync_baseline(self) -> bool:
        return bool(self._pending_full_sync_names)

    def clear_pending_full_sync_baseline(self) -> None:
        self._pending_full_sync_names.clear()

    def prewarm_baseline_from_metadata(self, metadata: TensorMetadata) -> None:
        if self.should_prewarm_baseline():
            for name, (shape, dtype) in metadata.items():
                self._ensure_baseline(name, tuple(int(dim) for dim in shape), dtype)

    def record_sparse_delta_range(
        self,
        name: str,
        *,
        start: int,
        values: torch.Tensor,
    ) -> bool:
        baseline = self.baseline.get(name)
        if baseline is None:
            return False
        values = values.detach().to(device="cpu", dtype=self.delta_dtype)
        self._pending_updates[name] = _PendingSparseUpdate(
            locations=None,
            values=values,
            range_start=int(start),
        )
        return True

    def record_sparse_delta_locations(
        self,
        name: str,
        *,
        locations: torch.Tensor,
        values: torch.Tensor,
    ) -> bool:
        baseline = self.baseline.get(name)
        if baseline is None:
            return False
        if int(locations.numel()) != int(values.numel()):
            raise ValueError(
                f"Sparse update for {name!r} has {locations.numel()} locations "
                f"for {values.numel()} values."
            )
        locations = locations.detach().to(device="cpu", dtype=torch.long)
        values = values.detach().to(device="cpu", dtype=self.delta_dtype)
        self._pending_updates[name] = _PendingSparseUpdate(
            locations=locations,
            values=values,
        )
        return True

    def prepare_sparse_delta_payload(
        self,
        tensors: TensorBatch,
        *,
        target_device: torch.device | None = None,
    ) -> tuple[bool, TensorPayload | TensorBatch]:
        payload_device = target_device or torch.device("cpu")
        if not tensors:
            return True, sparse_codec.empty_sparse_indices_payload(
                payload_device, self.delta_dtype
            )
        if not self.is_delta_sync():
            self._mark_pending_full_sync(tensors)
            return False, tensors
        if self._pending_full_sync_names:
            sample = next(iter(self._pending_full_sync_names))
            raise RuntimeError(
                "Delta baseline snapshot is still pending after the last full sync "
                f"(first pending tensor: {sample!r})."
            )
        if self._pending_updates:
            return self._prepare_recorded_sparse_delta_payload(
                tensors,
                target_device=target_device,
            )

        sparse_infos = []
        for name, tensor in tensors:
            baseline = self.baseline.get(name)
            if (
                baseline is None
                or tuple(baseline.shape) != tuple(tensor.shape)
                or baseline.dtype != tensor.dtype
            ):
                self._mark_pending_full_sync([(name, tensor)])
                return False, [(name, tensor)]
            current_cpu = tensor.detach().to(
                device="cpu", dtype=tensor.dtype, copy=True
            )
            current_flat = current_cpu.reshape(-1)
            base_flat = baseline.reshape(-1)
            current_flat.sub_(base_flat)
            locations = torch.nonzero(current_flat, as_tuple=False).reshape(-1)
            values = current_flat.index_select(0, locations).to(dtype=self.delta_dtype)
            if locations.numel():
                sparse_infos.append((name, current_cpu, locations, values))
                self._pending_updates[name] = _PendingSparseUpdate(
                    locations=locations,
                    values=values,
                )

        if not sparse_infos:
            return True, sparse_codec.empty_sparse_indices_payload(
                payload_device,
                self.delta_dtype,
            )
        payload = sparse_codec.encode_sparse_infos(
            sparse_infos,
            index_encoding=self.index_encoding,
        )
        if target_device is not None:
            payload = self._move_payload(payload, target_device)
        return True, payload

    def on_sync_succeeded(self) -> None:
        for name, update in self._pending_updates.items():
            self._apply_pending_sparse_update(name, update)
        self._pending_updates.clear()
        self._pending_full_sync_names.clear()
        self.committed_syncs += 1

    def on_sync_failed(self) -> None:
        self._pending_updates.clear()
        self._pending_full_sync_names.clear()

    def snapshot_pending_full_sync_baseline(
        self,
        tensors: Iterable[NamedTensor],
    ) -> None:
        for name, tensor in tensors:
            if (
                not self._pending_full_sync_names
                or name in self._pending_full_sync_names
            ):
                self._copy_to_baseline(
                    name,
                    tensor.detach().to(device="cpu", dtype=tensor.dtype, copy=True),
                )
                self._pending_full_sync_names.discard(name)

    def _mark_pending_full_sync(self, tensors: TensorBatch) -> None:
        if self.full_sync_interval <= 1:
            return
        for name, tensor in tensors:
            self._ensure_baseline(
                name,
                tuple(int(dim) for dim in tensor.shape),
                tensor.dtype,
            )
            self._pending_full_sync_names.add(name)

    def _copy_to_baseline(self, name: str, tensor: torch.Tensor) -> None:
        baseline = self._ensure_baseline(
            name,
            tuple(int(dim) for dim in tensor.shape),
            tensor.dtype,
        )
        baseline.copy_(tensor.reshape(baseline.shape))

    def _apply_pending_sparse_update(
        self, name: str, update: _PendingSparseUpdate
    ) -> None:
        baseline = self.baseline[name].reshape(-1)
        values = update.values.to(dtype=baseline.dtype)
        if update.range_start is not None:
            locations = None
            target = baseline.narrow(0, update.range_start, int(values.numel()))
        else:
            if update.locations is None:
                raise RuntimeError(f"Sparse update for {name!r} has no locations.")
            locations = update.locations.to(dtype=torch.long)
            target = (
                baseline.narrow(0, int(locations[0]), int(locations.numel()))
                if sparse_codec.is_contiguous_range(locations)
                else None
            )
        if target is not None:
            target.add_(values)
        else:
            if locations is None:
                raise RuntimeError(f"Sparse update for {name!r} has no locations.")
            updated_values = baseline.index_select(0, locations).add_(values)
            baseline.index_copy_(0, locations, updated_values)

    def _prepare_recorded_sparse_delta_payload(
        self,
        tensors: TensorBatch,
        *,
        target_device: torch.device | None,
    ) -> tuple[bool, TensorPayload | TensorBatch]:
        sparse_infos = []
        for name, tensor in tensors:
            update = self._pending_updates.get(name)
            if update is None:
                continue
            baseline = self.baseline.get(name)
            if (
                baseline is None
                or tuple(baseline.shape) != tuple(tensor.shape)
                or baseline.dtype != tensor.dtype
            ):
                self._mark_pending_full_sync([(name, tensor)])
                return False, [(name, tensor)]
            if update.range_start is not None:
                sparse_infos.append((name, baseline, update.range_start, update.values))
            else:
                if update.locations is None:
                    raise RuntimeError(f"Sparse update for {name!r} has no locations.")
                sparse_infos.append((name, baseline, update.locations, update.values))
        if not sparse_infos:
            return True, sparse_codec.empty_sparse_indices_payload(
                target_device or torch.device("cpu"), self.delta_dtype
            )
        payload = sparse_codec.encode_sparse_infos(
            sparse_infos,
            index_encoding=self.index_encoding,
        )
        if target_device is not None:
            payload = self._move_payload(payload, target_device)
        return True, payload

    def _ensure_baseline(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        current = self.baseline.get(name)
        if (
            current is not None
            and tuple(current.shape) == shape
            and current.dtype == dtype
        ):
            return current
        old = self._entries.pop(name, None)
        if old is not None and old.path:
            try:
                os.unlink(old.path)
            except FileNotFoundError:
                pass
        numel = 1
        for dim in shape:
            numel *= dim
        if self.baseline_in_memory:
            baseline = torch.empty(shape, dtype=dtype, device="cpu")
            self.baseline[name] = baseline
            self._entries[name] = _BaselineEntry(baseline, None)
            return baseline
        directory = self.baseline_mmap_dir or tempfile.gettempdir()
        fd, path = tempfile.mkstemp(prefix="nrl-refit-baseline-", dir=directory)
        os.close(fd)
        with open(path, "wb") as handle:
            handle.truncate(numel * dtype_itemsize(dtype))
        baseline = torch.from_file(path, shared=True, size=numel, dtype=dtype).view(
            shape
        )
        self.baseline[name] = baseline
        self._entries[name] = _BaselineEntry(baseline, path)
        return baseline

    @staticmethod
    def _move_payload(payload: TensorPayload, device: torch.device) -> TensorPayload:
        tensors, transport, metadata = payload
        return (
            [
                (name, tensor.to(device=device, non_blocking=True))
                for name, tensor in tensors
            ],
            transport,
            metadata,
        )


def create_vllm_delta_transfer_tracker(
    generation_config: Mapping[str, Any] | None,
) -> DeltaCompressionTracker | None:
    if not generation_config:
        return None
    delta_config = generation_config.get("delta_compression")
    if delta_config is not None:
        delta_config = config_to_dict(delta_config)
    if not delta_config or not delta_config.get("enabled", False):
        return None
    return DeltaCompressionTracker(delta_config)
