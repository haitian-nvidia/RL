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

import contextlib
import json
import os
from collections.abc import Callable, Iterable, Iterator, Mapping
from functools import lru_cache
from typing import Any, Literal, cast

import torch

G_PAYLOAD_ALIGNMENT_BYTES = 8
G_PACKED_INDICES_NAME = "__packed_indices__"
G_PACKED_VALUES_NAME = "__packed_values__"
G_INDEX_START_KEY = "index_start"
G_INDEX_END_KEY = "index_end"

DeltaCompressionTransport = Literal["dense", "sparse_indices"]
WeightTransferKind = Literal["full", "delta", "done"]
G_DENSE_TRANSPORT: DeltaCompressionTransport = "dense"
G_SPARSE_INDICES_TRANSPORT: DeltaCompressionTransport = "sparse_indices"
G_FULL_UPDATE_KIND: WeightTransferKind = "full"
G_DELTA_UPDATE_KIND: WeightTransferKind = "delta"
G_TRANSFER_DONE_KIND: WeightTransferKind = "done"

NamedTensor = tuple[str, torch.Tensor]
TensorBatch = list[NamedTensor]
WeightLoadFunc = Callable[[TensorBatch], None]
TensorPayload = tuple[TensorBatch, DeltaCompressionTransport, list[dict[str, Any]]]
PayloadEvents = tuple[torch.cuda.Event, ...]
QueuedPayload = tuple[WeightTransferKind, TensorPayload, PayloadEvents]
SparseBucketPayload = tuple[TensorPayload, PayloadEvents]
HeaderRefs = tuple[torch.Tensor, torch.Tensor | None]
TensorMetadata = Mapping[str, tuple[Iterable[int], torch.dtype]]

G_REFIT_PREWARM_DELTA_BASELINE_ENV = "NRL_REFIT_PREWARM_DELTA_BASELINE"
G_REFIT_BASELINE_IN_MEMORY_ENV = "NRL_REFIT_BASELINE_IN_MEMORY"
G_REFIT_BASELINE_MMAP_DIR_ENV = "NRL_REFIT_BASELINE_MMAP_DIR"
G_REFIT_DIRECT_SPARSE_VLLM_LOAD_ENV = "NRL_REFIT_DIRECT_SPARSE_VLLM_LOAD"

G_TENSOR_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float64": torch.float64,
    "double": torch.float64,
    "int64": torch.int64,
    "long": torch.int64,
    "int32": torch.int32,
    "int": torch.int32,
    "int16": torch.int16,
    "short": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}

for _float8_dtype_name in (
    "float8_e4m3fn",
    "float8_e5m2",
    "float8_e4m3fnuz",
    "float8_e5m2fnuz",
):
    _float8_dtype = getattr(torch, _float8_dtype_name, None)
    if _float8_dtype is not None:
        G_TENSOR_DTYPE_MAP[_float8_dtype_name] = _float8_dtype
del _float8_dtype, _float8_dtype_name


def env_flag(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    return (
        default
        if value is None
        else value.strip().lower() in {"1", "true", "yes", "on"}
    )


def env_int(name: str, *, default: int, min_value: int | None = None) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        parsed = default
    else:
        try:
            parsed = int(value)
        except ValueError:
            raise ValueError(f"Expected integer value for {name}.") from None
    if min_value is not None and parsed < min_value:
        raise ValueError(f"{name} must be >= {min_value}.")
    return parsed


def config_to_dict(config: Any | None) -> dict[str, Any]:
    if config is None:
        return {}
    if hasattr(config, "model_dump"):
        return dict(config.model_dump())
    return dict(config)


def config_env_flag(
    config: Mapping[str, Any],
    key: str,
    *,
    env_name: str,
    default: bool,
) -> bool:
    if key in config:
        return bool(config[key])
    return env_flag(env_name, default=default)


def is_refit_receiver_timing(key: str, value: Any) -> bool:
    return (
        key.startswith("receiver_")
        and key.endswith("_s")
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    )


def dtype_itemsize(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def dtype_to_name(dtype: torch.dtype) -> str:
    return str(dtype).split(".", 1)[-1]


def dtype_from_name(name: str) -> torch.dtype:
    try:
        return G_TENSOR_DTYPE_MAP[name]
    except KeyError:
        raise ValueError(f"Unsupported tensor dtype {name!r}") from None


@lru_cache(maxsize=1)
def get_target_packed_tensor_size() -> int:
    memory_ratio = os.getenv("NRL_REFIT_BUFFER_MEMORY_RATIO", "0.02")
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    return min(int(props.total_memory * float(memory_ratio)), 5 * 1024**3)


@lru_cache(maxsize=1)
def get_num_buffers() -> int:
    return int(os.getenv("NRL_REFIT_NUM_BUFFERS", "2"))


def pack_named_tensors(tensors: TensorBatch) -> tuple[torch.Tensor, list[dict]]:
    chunks = []
    entries = []
    for name, tensor in tensors:
        tensor = tensor.contiguous()
        byte_view = tensor.view(torch.uint8).view(-1)
        byte_size = int(byte_view.numel())
        pad = (-byte_size) % G_PAYLOAD_ALIGNMENT_BYTES
        chunks.append(byte_view)
        if pad:
            chunks.append(torch.zeros(pad, dtype=torch.uint8, device=tensor.device))
        entries.append(
            {
                "name": name,
                "shape": [int(dim) for dim in tensor.shape],
                "dtype": dtype_to_name(tensor.dtype),
                "byte_size": byte_size,
                "wire_byte_size": byte_size + pad,
            }
        )
    if not chunks:
        return torch.empty(0, dtype=torch.uint8), []
    return torch.cat(chunks, dim=0), entries


def unpack_named_tensors(
    payload: torch.Tensor,
    entries: list[dict[str, Any]],
) -> TensorBatch:
    byte_views = payload.split_with_sizes(
        [int(entry["wire_byte_size"]) for entry in entries]
    )
    return [
        (
            str(entry["name"]),
            byte_view[: int(entry["byte_size"])]
            .view(dtype_from_name(str(entry["dtype"])))
            .view(tuple(int(dim) for dim in entry["shape"])),
        )
        for entry, byte_view in zip(entries, byte_views, strict=True)
    ]


def wire_bytes(tensors: TensorBatch) -> int:
    return sum(
        (byte_size := int(tensor.numel() * tensor.element_size()))
        + (-byte_size) % G_PAYLOAD_ALIGNMENT_BYTES
        for _, tensor in tensors
    )


@contextlib.contextmanager
def additive_weight_load_context(target_tensors: Iterable[torch.Tensor]):
    """Make vLLM loader copy/fill calls add deltas into existing tensors."""
    target_storage_ptrs = {
        ptr
        for tensor in target_tensors
        if (ptr := _tensor_storage_ptr(tensor)) is not None
    }
    old_copy = torch.Tensor.copy_
    old_fill = torch.Tensor.fill_

    def copy_add(self, src, non_blocking=False):
        if _tensor_storage_ptr(self) in target_storage_ptrs and torch.is_tensor(src):
            return self.add_(src.to(device=self.device, dtype=self.dtype), alpha=1)
        return old_copy(self, src, non_blocking=non_blocking)

    def fill_add(self, value, *args, **kwargs):
        if _tensor_storage_ptr(self) in target_storage_ptrs:
            return self.add_(value)
        return old_fill(self, value, *args, **kwargs)

    torch.Tensor.copy_ = copy_add  # pyrefly: ignore[bad-assignment]
    torch.Tensor.fill_ = fill_add  # pyrefly: ignore[bad-assignment]
    try:
        yield
    finally:
        torch.Tensor.copy_ = old_copy
        torch.Tensor.fill_ = old_fill


def _tensor_storage_ptr(tensor: torch.Tensor) -> int | None:
    try:
        return int(tensor.untyped_storage().data_ptr())
    except RuntimeError:
        return None


def _wire_bytes(item: NamedTensor) -> int:
    tensor = item[1]
    return int(tensor.numel() * tensor.element_size())


def next_chunk(
    iterator: Iterator[NamedTensor],
    target_chunk_bytes: int,
    *,
    pending_item: NamedTensor | None = None,
) -> tuple[TensorBatch, NamedTensor | None]:
    chunk: TensorBatch = []
    chunk_bytes = 0
    if pending_item is not None:
        chunk.append(pending_item)
        chunk_bytes = _wire_bytes(pending_item)
        pending_item = None
    for item in iterator:
        item_bytes = _wire_bytes(item)
        if chunk and chunk_bytes + item_bytes > target_chunk_bytes:
            return chunk, item
        chunk.append(item)
        chunk_bytes += item_bytes
        if chunk_bytes >= target_chunk_bytes:
            break
    return chunk, None


def target_chunk_size() -> int:
    if torch.cuda.is_available():
        return int(get_target_packed_tensor_size())
    return int(os.getenv("NRL_REFIT_CPU_TARGET_PACKED_TENSOR_SIZE", 64 * 1024**2))


def broadcast_header(
    header: Mapping[str, Any],
    *,
    group: Any,
    src: int,
    device: torch.device | int | str,
) -> tuple[dict[str, Any], HeaderRefs]:
    is_src = int(group.rank) == src
    encoded = _encode_header_metadata(header)
    if is_src:
        control_tensor = torch.tensor(
            _header_control_values(header, len(encoded)),
            dtype=torch.int64,
            device=device,
        )
    else:
        control_tensor = torch.empty(4, dtype=torch.int64, device=device)
    group.broadcast(control_tensor, src=src)

    # The source already knows every header value, so skip the device->host
    # readback of the control tensor and only decode it on receivers.
    if is_src:
        metadata_len = len(encoded)
    else:
        kind, transport, payload_numel, metadata_len = _decode_header_control(
            control_tensor
        )

    metadata_tensor = None
    metadata: dict[str, Any] = {}
    if metadata_len > 0:
        if is_src:
            metadata_tensor = torch.tensor(
                list(encoded), dtype=torch.uint8, device=device
            )
        else:
            metadata_tensor = torch.empty(
                metadata_len, dtype=torch.uint8, device=device
            )
        group.broadcast(metadata_tensor, src=src)
        if not is_src:
            metadata = json.loads(
                metadata_tensor.cpu().numpy().tobytes().decode("utf-8")
            )

    if is_src:
        received_header = dict(header)
    else:
        received_header = {
            "kind": kind,
            "transport": transport,
            "payload_entries": [],
            "payload_numel": payload_numel,
            "sparse_metadata": [],
        }
        received_header.update(metadata)
    return received_header, (control_tensor, metadata_tensor)


def _header_control_values(header: Mapping[str, Any], metadata_len: int) -> list[int]:
    kind_code = {
        G_TRANSFER_DONE_KIND: 0,
        G_FULL_UPDATE_KIND: 1,
        G_DELTA_UPDATE_KIND: 2,
    }.get(header["kind"])
    if kind_code is None:
        raise ValueError(f"Unsupported weight transfer header kind: {header['kind']!r}")

    transport_code = {
        G_DENSE_TRANSPORT: 0,
        G_SPARSE_INDICES_TRANSPORT: 1,
    }.get(header.get("transport", G_DENSE_TRANSPORT))
    if transport_code is None:
        raise ValueError(
            f"Unsupported weight transfer header transport: {header.get('transport')!r}"
        )
    return [
        kind_code,
        transport_code,
        int(header.get("payload_numel", 0)),
        metadata_len,
    ]


def _decode_header_control(
    control_tensor: torch.Tensor,
) -> tuple[WeightTransferKind, str, int, int]:
    kind_code, transport_code, payload_numel, metadata_len = [
        int(value) for value in control_tensor.cpu().tolist()
    ]
    kind_by_code: dict[int, WeightTransferKind] = {
        0: G_TRANSFER_DONE_KIND,
        1: G_FULL_UPDATE_KIND,
        2: G_DELTA_UPDATE_KIND,
    }
    transport_by_code: dict[int, DeltaCompressionTransport] = {
        0: G_DENSE_TRANSPORT,
        1: G_SPARSE_INDICES_TRANSPORT,
    }
    try:
        kind = kind_by_code[kind_code]
        transport = transport_by_code[transport_code]
    except KeyError:
        raise ValueError(
            "Unsupported weight transfer header control values: "
            f"kind={kind_code}, transport={transport_code}"
        ) from None
    return kind, transport, payload_numel, metadata_len


def _encode_header_metadata(header: Mapping[str, Any]) -> bytes:
    metadata = {
        key: header[key]
        for key in ("payload_entries", "sparse_metadata", "is_delta_sync")
        if header.get(key)
    }
    return json.dumps(metadata).encode("utf-8") if metadata else b""


def recv_payload(
    payload_numel: int,
    *,
    group: Any,
    src: int,
    device: torch.device | int | str,
) -> torch.Tensor:
    payload = torch.empty(payload_numel, dtype=torch.uint8, device=device)
    if payload.numel() > 0:
        group.broadcast(payload, src=src)
    return payload


def cuda_streams(
    device: torch.device | int | str | None = None,
) -> list[torch.cuda.Stream | None]:
    if not torch.cuda.is_available():
        return [None]
    normalized_device = normalize_device(device)
    if normalized_device is not None and normalized_device.type != "cuda":
        return [None]
    with cuda_device(normalized_device):
        return [torch.cuda.Stream() for _ in range(get_num_buffers())]


@contextlib.contextmanager
def use_stream(streams: list[torch.cuda.Stream | None], index: int):
    stream = streams[index]
    if stream is None:
        yield
        return
    with torch.cuda.stream(stream):
        yield


def record_header_stream(refs: HeaderRefs) -> None:
    control_tensor, metadata_tensor = refs
    record_tensor_stream(control_tensor)
    if metadata_tensor is not None:
        record_tensor_stream(metadata_tensor)


def record_tensor_stream(tensor: torch.Tensor) -> None:
    if tensor.is_cuda:
        tensor.record_stream(torch.cuda.current_stream())


def sync_streams(streams: list[torch.cuda.Stream | None]) -> None:
    for stream in streams:
        if stream is not None:
            stream.synchronize()


def synchronize_current_transfer_stream(device: torch.device | int | str) -> None:
    if not torch.cuda.is_available():
        return
    normalized_device = normalize_device(device)
    if normalized_device is None or normalized_device.type == "cuda":
        torch.cuda.current_stream(normalized_device).synchronize()


def record_stream_event(stream: torch.cuda.Stream | None) -> torch.cuda.Event | None:
    if stream is None:
        return None
    return cast(torch.cuda.Event, stream.record_event())


def record_payload_readiness_events() -> PayloadEvents:
    if not torch.cuda.is_available():
        return ()
    return (cast(torch.cuda.Event, torch.cuda.current_stream().record_event()),)


def wait_for_payload_events(events: Iterable[torch.cuda.Event | None]) -> None:
    seen_events: set[int] = set()
    current_stream: torch.cuda.Stream | None = None
    for event in events:
        if event is None or id(event) in seen_events:
            continue
        if current_stream is None:
            current_stream = torch.cuda.current_stream()
        current_stream.wait_event(event)
        seen_events.add(id(event))


@contextlib.contextmanager
def cuda_device(device: torch.device | None):
    if device is None or device.type != "cuda":
        yield
        return
    with torch.cuda.device(device):
        yield


def normalize_device(device: torch.device | int | str | None) -> torch.device | None:
    if device is None:
        return None
    if isinstance(device, int):
        return torch.device("cuda", device)
    return torch.device(device)
