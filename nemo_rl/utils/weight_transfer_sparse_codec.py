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

import threading
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import numpy as np
import torch

from nemo_rl.utils.weight_transfer_protocol import (
    G_INDEX_END_KEY,
    G_INDEX_START_KEY,
    G_PACKED_INDICES_NAME,
    G_PACKED_VALUES_NAME,
    G_SPARSE_INDICES_TRANSPORT,
    TensorBatch,
    TensorPayload,
    dtype_from_name,
    dtype_to_name,
)

SparseMetadataSpan = tuple[int, int, int, int]
G_INDEX_ENCODING_KEY = "index_encoding"
G_INDEX_ENCODING_DELTAS = "deltas"
G_INDEX_ENCODING_DELTAS_ZSTD = "deltas_zstd"
G_INDEX_ENCODING_INDICES = "indices"
G_INDEX_ENCODING_RANGE = "range"
G_INDEX_UNCOMPRESSED_BYTES_KEY = "index_uncompressed_bytes"
G_INDEX_WIDTH_KEY = "index_width"
G_EXPLICIT_INDEX_WIDTH_KEY = "explicit_index_width"
G_RANGE_START_KEY = "range_start"
G_SUPPORTED_INDEX_ENCODINGS = {
    G_INDEX_ENCODING_INDICES,
    G_INDEX_ENCODING_DELTAS,
    G_INDEX_ENCODING_DELTAS_ZSTD,
}
_ZSTD_LOCAL = threading.local()


def sparse_metadata_spans(
    metadata: Sequence[dict[str, Any]],
) -> tuple[SparseMetadataSpan, ...]:
    spans = []
    for item in metadata:
        index_end = int(item[G_INDEX_END_KEY])
        value_end = int(item["value_end"])
        spans.append(
            (
                int(item[G_INDEX_START_KEY]),
                index_end,
                int(item["value_start"]),
                value_end,
            )
        )
    return tuple(spans)


def empty_sparse_indices_payload(
    device: torch.device | int | str,
    dtype: torch.dtype,
) -> TensorPayload:
    return (
        [
            (G_PACKED_INDICES_NAME, torch.empty(0, dtype=torch.int32, device=device)),
            (G_PACKED_VALUES_NAME, torch.empty(0, dtype=dtype, device=device)),
        ],
        G_SPARSE_INDICES_TRANSPORT,
        [],
    )


def validate_sparse_index_encoding(index_encoding: str) -> str:
    normalized = index_encoding.lower()
    if normalized not in G_SUPPORTED_INDEX_ENCODINGS:
        raise ValueError(
            f"Unsupported sparse index encoding {index_encoding!r}; "
            f"expected one of {sorted(G_SUPPORTED_INDEX_ENCODINGS)}."
        )
    if normalized == G_INDEX_ENCODING_DELTAS_ZSTD:
        _require_zstandard()
    return normalized


def encode_sparse_infos(
    infos: Iterable[tuple[str, torch.Tensor, torch.Tensor | int, torch.Tensor]],
    *,
    index_encoding: str = G_INDEX_ENCODING_INDICES,
) -> TensorPayload:
    index_encoding = validate_sparse_index_encoding(index_encoding)
    packed_locations = []
    packed_values = []
    metadata: list[dict[str, Any]] = []
    index_offset = value_offset = 0
    device: torch.device | None = None
    value_dtype: torch.dtype | None = None
    for name, tensor, raw_locations, raw_values in infos:
        values = raw_values.to(
            tensor.device if tensor.device.type != "cpu" else torch.device("cpu")
        )
        device = (
            raw_locations.device
            if isinstance(raw_locations, torch.Tensor) and device is None
            else device
        )
        device = values.device if device is None else device
        value_dtype = values.dtype if value_dtype is None else value_dtype
        count = int(values.numel())
        if isinstance(raw_locations, int):
            is_range = True
            range_start = raw_locations
            location_tensor = None
            location_metadata = {}
        else:
            if int(raw_locations.numel()) != count:
                raise ValueError(
                    f"Sparse tensor {name!r} has {raw_locations.numel()} locations "
                    f"for {count} values."
                )
            is_range = is_contiguous_range(raw_locations)
            if is_range:
                range_start = int(raw_locations[0])
                location_tensor = None
                location_metadata = {}
            else:
                location_tensor, location_metadata = _encode_explicit_locations(
                    raw_locations,
                    index_encoding=index_encoding,
                )
                packed_locations.append(location_tensor.to(device=device))
        index_count = 0 if location_tensor is None else int(location_tensor.numel())
        packed_values.append(values.to(device=device, dtype=value_dtype))
        item = {
            "name": name,
            "shape": tuple(int(dim) for dim in tensor.shape),
            "dtype": dtype_to_name(tensor.dtype),
            "numel": int(tensor.numel()),
            G_INDEX_START_KEY: index_offset,
            G_INDEX_END_KEY: index_offset + index_count,
            "value_start": value_offset,
            "value_end": value_offset + count,
        }
        if is_range:
            item[G_INDEX_ENCODING_KEY] = G_INDEX_ENCODING_RANGE
            item[G_RANGE_START_KEY] = range_start
        else:
            item.update(location_metadata)
        metadata.append(item)
        index_offset += index_count
        value_offset += count
    device = device or torch.device("cpu")
    value_dtype = value_dtype or torch.float32
    return (
        [
            (
                G_PACKED_INDICES_NAME,
                _cat_packed_index_parts(packed_locations, metadata, device)
                if packed_locations
                else torch.empty(0, dtype=torch.int32, device=device),
            ),
            (
                G_PACKED_VALUES_NAME,
                torch.cat(packed_values)
                if packed_values
                else torch.empty(0, dtype=value_dtype, device=device),
            ),
        ],
        G_SPARSE_INDICES_TRANSPORT,
        metadata,
    )


def packed_sparse_payload_tensors(
    payload_tensors: TensorBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(payload_tensors) >= 2 and (
        payload_tensors[0][0],
        payload_tensors[1][0],
    ) == (G_PACKED_INDICES_NAME, G_PACKED_VALUES_NAME):
        return payload_tensors[0][1], payload_tensors[1][1]
    raise KeyError("Sparse payload is missing packed locations or values")


def merge_sparse_payloads(payloads: Sequence[TensorPayload]) -> TensorPayload:
    if not payloads:
        return empty_sparse_indices_payload(torch.device("cpu"), torch.float32)

    packed_parts = []
    value_parts = []
    metadata: list[dict[str, Any]] = []
    index_offset = value_offset = 0
    for tensors, transport, sparse_metadata in payloads:
        if transport != G_SPARSE_INDICES_TRANSPORT:
            raise ValueError(f"Cannot merge sparse payload transport={transport!r}.")
        packed_indices, packed_values = packed_sparse_payload_tensors(tensors)
        packed_parts.append(packed_indices)
        value_parts.append(packed_values)
        for item in sparse_metadata:
            merged_item = dict(item)
            merged_item[G_INDEX_START_KEY] = int(item[G_INDEX_START_KEY]) + index_offset
            merged_item[G_INDEX_END_KEY] = int(item[G_INDEX_END_KEY]) + index_offset
            merged_item["value_start"] = int(item["value_start"]) + value_offset
            merged_item["value_end"] = int(item["value_end"]) + value_offset
            metadata.append(merged_item)
        index_offset += int(packed_indices.numel())
        value_offset += int(packed_values.numel())

    packed_parts = _normalize_packed_index_parts(packed_parts)
    _update_explicit_index_width_metadata(metadata, packed_parts)
    return (
        [
            (G_PACKED_INDICES_NAME, torch.cat(packed_parts, dim=0)),
            (G_PACKED_VALUES_NAME, torch.cat(value_parts, dim=0)),
        ],
        G_SPARSE_INDICES_TRANSPORT,
        metadata,
    )


def decode_sparse(
    payload_tensors: TensorBatch,
    metadata: Sequence[dict[str, Any]],
    device: torch.device | int | str,
    byte_cap: int,
) -> Iterator[TensorBatch]:
    raw_locations, raw_values = packed_sparse_payload_tensors(payload_tensors)
    packed_values = raw_values.to(device=device, non_blocking=True)
    batch: TensorBatch = []
    batch_bytes = 0
    for item, span in zip(metadata, sparse_metadata_spans(metadata), strict=True):
        _, _, value_start, value_end = span
        dtype = dtype_from_name(str(item["dtype"]))
        tensor = torch.zeros(int(item["numel"]), dtype=dtype, device=device)
        values = packed_values[value_start:value_end].to(dtype=dtype)
        locations = sparse_locations_for_item(
            item,
            raw_locations,
            span,
            device=device,
            dtype=torch.long,
        )
        tensor.index_copy_(0, locations, values)
        tensor = tensor.view(tuple(int(dim) for dim in item["shape"]))
        tensor_bytes = int(tensor.numel() * tensor.element_size())
        if batch and batch_bytes + tensor_bytes > byte_cap:
            yield batch
            batch = []
            batch_bytes = 0
        batch.append((str(item["name"]), tensor))
        batch_bytes += tensor_bytes
    if batch:
        yield batch


def sparse_locations_for_item(
    item: dict[str, Any],
    packed_locations: torch.Tensor,
    span: SparseMetadataSpan,
    *,
    device: torch.device | int | str,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    index_start, index_end, _, _ = span
    sparse_range = sparse_range_for_item(item, span)
    if sparse_range is not None:
        start, count = sparse_range
        return torch.arange(start, start + count, device=device, dtype=dtype)
    index_encoding = sparse_index_encoding_for_item(item)
    if index_encoding == G_INDEX_ENCODING_INDICES:
        explicit_width = item.get(G_EXPLICIT_INDEX_WIDTH_KEY)
        if explicit_width is not None and int(explicit_width) not in (4, 8):
            raise ValueError(
                f"Unsupported sparse explicit index width {explicit_width}."
            )
        return packed_locations[index_start:index_end].to(device=device, dtype=dtype)
    if index_encoding in (G_INDEX_ENCODING_DELTAS, G_INDEX_ENCODING_DELTAS_ZSTD):
        return _decode_delta_locations(
            packed_locations[index_start:index_end],
            item,
            device=device,
            dtype=dtype,
        )
    raise ValueError(f"Unsupported sparse index encoding {index_encoding!r}.")


def sparse_index_encoding_for_item(item: dict[str, Any]) -> str:
    index_encoding = item.get(G_INDEX_ENCODING_KEY, G_INDEX_ENCODING_INDICES)
    if index_encoding == G_INDEX_ENCODING_INDICES and G_INDEX_WIDTH_KEY in item:
        return (
            G_INDEX_ENCODING_DELTAS_ZSTD
            if G_INDEX_UNCOMPRESSED_BYTES_KEY in item
            else G_INDEX_ENCODING_DELTAS
        )
    return str(index_encoding)


def sparse_range_for_item(
    item: dict[str, Any],
    span: SparseMetadataSpan,
) -> tuple[int, int] | None:
    if item.get(G_INDEX_ENCODING_KEY) != G_INDEX_ENCODING_RANGE:
        return None
    _, _, value_start, value_end = span
    return int(item[G_RANGE_START_KEY]), value_end - value_start


def is_contiguous_range(locations: torch.Tensor) -> bool:
    count = int(locations.numel())
    if count == 0:
        return False
    return count == 1 or int(locations[-1] - locations[0] + 1) == count


def _encode_explicit_locations(
    locations: torch.Tensor,
    *,
    index_encoding: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if index_encoding == G_INDEX_ENCODING_INDICES:
        detached = locations.detach()
        if int(detached.numel()) == 0:
            max_location = 0
            min_location = 0
        else:
            min_location = int(detached.min().item())
            max_location = int(detached.max().item())
        if min_location < 0:
            raise ValueError("Sparse explicit indices must be non-negative.")
        index_dtype = (
            torch.int64 if max_location > torch.iinfo(torch.int32).max else torch.int32
        )
        return (
            locations.to(dtype=index_dtype),
            {
                G_INDEX_ENCODING_KEY: G_INDEX_ENCODING_INDICES,
                G_EXPLICIT_INDEX_WIDTH_KEY: 8 if index_dtype == torch.int64 else 4,
            },
        )

    deltas = _location_deltas_cpu(locations)
    max_delta = int(deltas.max()) if deltas.size else 0
    width = 2 if max_delta <= 65535 else 4
    dtype = np.uint16 if width == 2 else np.uint32
    raw = deltas.astype(dtype, copy=False).tobytes()
    metadata = {
        G_INDEX_ENCODING_KEY: index_encoding,
        G_INDEX_WIDTH_KEY: width,
    }
    if index_encoding == G_INDEX_ENCODING_DELTAS_ZSTD:
        metadata[G_INDEX_UNCOMPRESSED_BYTES_KEY] = len(raw)
        raw = _zstd_compress(raw)
    return torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy()), metadata


def _location_deltas_cpu(locations: torch.Tensor) -> np.ndarray:
    local_idx = locations.detach().cpu().numpy().astype(np.int64, copy=False)
    if local_idx.size == 0:
        return local_idx.astype(np.uint16, copy=False)
    prev = np.empty_like(local_idx)
    prev[0] = -1
    prev[1:] = local_idx[:-1]
    deltas = local_idx - prev - 1
    if np.any(deltas < 0):
        raise ValueError("Sparse delta index encoding requires sorted locations.")
    return deltas


def _decode_delta_locations(
    packed_locations: torch.Tensor,
    item: dict[str, Any],
    *,
    device: torch.device | int | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    raw = packed_locations.detach().cpu().numpy().astype(np.uint8, copy=False).tobytes()
    index_encoding = item.get(G_INDEX_ENCODING_KEY)
    if index_encoding == G_INDEX_ENCODING_DELTAS_ZSTD:
        raw = _zstd_decompress(
            raw,
            max_output_size=int(item[G_INDEX_UNCOMPRESSED_BYTES_KEY]),
        )
    width = int(item[G_INDEX_WIDTH_KEY])
    if width == 2:
        deltas = np.frombuffer(raw, dtype=np.uint16).astype(np.int64, copy=False)
    elif width == 4:
        deltas = np.frombuffer(raw, dtype=np.uint32).astype(np.int64, copy=False)
    else:
        raise ValueError(f"Unsupported sparse delta index width {width}.")
    locations = np.cumsum(deltas + 1, dtype=np.int64) - 1
    return torch.from_numpy(locations).to(device=device, dtype=dtype)


def _normalize_packed_index_parts(parts: list[torch.Tensor]) -> list[torch.Tensor]:
    dtypes = {part.dtype for part in parts if int(part.numel()) > 0}
    if not dtypes:
        dtype = parts[0].dtype if parts else torch.int32
    elif len(dtypes) == 1:
        dtype = next(iter(dtypes))
    elif dtypes <= {torch.int32, torch.int64}:
        dtype = torch.int64
    else:
        raise ValueError("Cannot merge sparse payloads with mixed index dtypes.")
    normalized = []
    for part in parts:
        normalized.append(part.to(dtype=dtype) if part.dtype != dtype else part)
    return normalized


def _cat_packed_index_parts(
    parts: list[torch.Tensor],
    metadata: list[dict[str, Any]],
    device: torch.device | int | str,
) -> torch.Tensor:
    parts = _normalize_packed_index_parts(parts)
    _update_explicit_index_width_metadata(metadata, parts)
    return torch.cat(parts).to(device=device)


def _update_explicit_index_width_metadata(
    metadata: list[dict[str, Any]],
    parts: list[torch.Tensor],
) -> None:
    dtype = next((part.dtype for part in parts if int(part.numel()) > 0), None)
    if dtype not in (torch.int32, torch.int64):
        return
    width = 8 if dtype == torch.int64 else 4
    for item in metadata:
        if item.get(G_INDEX_ENCODING_KEY) == G_INDEX_ENCODING_INDICES:
            item[G_EXPLICIT_INDEX_WIDTH_KEY] = width


def _require_zstandard():
    try:
        import zstandard
    except ImportError:
        raise RuntimeError(
            "Sparse index encoding 'deltas_zstd' requires the zstandard package."
        ) from None
    return zstandard


def _zstd_compress(raw: bytes) -> bytes:
    compressor = getattr(_ZSTD_LOCAL, "compressor", None)
    if compressor is None:
        compressor = _require_zstandard().ZstdCompressor(level=1)
        _ZSTD_LOCAL.compressor = compressor
    return compressor.compress(raw)


def _zstd_decompress(raw: bytes, *, max_output_size: int) -> bytes:
    decompressor = getattr(_ZSTD_LOCAL, "decompressor", None)
    if decompressor is None:
        decompressor = _require_zstandard().ZstdDecompressor()
        _ZSTD_LOCAL.decompressor = decompressor
    return decompressor.decompress(raw, max_output_size=max_output_size)
