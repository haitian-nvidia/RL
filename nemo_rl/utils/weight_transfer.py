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

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from nemo_rl.utils.weight_transfer_delta_tracker import DeltaCompressionTracker
from nemo_rl.utils.weight_transfer_payload_queue import PendingWirePayloads
from nemo_rl.utils.weight_transfer_protocol import (
    G_DELTA_UPDATE_KIND,
    G_FULL_UPDATE_KIND,
    G_TRANSFER_DONE_KIND,
    NamedTensor,
    TensorBatch,
    WeightLoadFunc,
    broadcast_header,
    cuda_streams,
    next_chunk,
    record_header_stream,
    record_stream_event,
    record_tensor_stream,
    recv_payload,
    sync_streams,
    synchronize_current_transfer_stream,
    target_chunk_size,
    unpack_named_tensors,
    use_stream,
)

SparseWeightLoadFunc = Callable[[TensorBatch, list[dict[str, Any]]], None]


@dataclass(frozen=True)
class WeightTransferResult:
    loaded_any: bool
    is_delta_sync: bool


def packed_weight_transfer_producer(
    iterator: Iterable[NamedTensor],
    *,
    group: Any,
    src: int,
    delta_tracker: DeltaCompressionTracker | None = None,
) -> None:
    """Broadcast dense or sparse-delta refit payloads over the update group."""
    transfer_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank = int(group.rank)
    if rank != src:
        _participate_in_source_iteration_and_drain(
            iterator=iter(iterator),
            group=group,
            src=src,
            device=transfer_device,
        )
        return

    encode_streams = cuda_streams()
    broadcast_streams = cuda_streams()
    buffer_idx = 0
    live_refs: list[Any] = [None for _ in broadcast_streams]
    chunk_size = target_chunk_size()
    tensor_iterator = iter(iterator)
    pending_item = None
    is_delta_sync = (
        delta_tracker.is_delta_sync() if delta_tracker is not None else False
    )
    pending_payloads = PendingWirePayloads(
        transfer_device=transfer_device,
        sparse_bucket_size_bytes=(
            delta_tracker.sparse_bucket_size_bytes if delta_tracker is not None else 1
        ),
        is_delta_sync=is_delta_sync,
    )

    def send(
        header: Mapping[str, Any],
        payload: torch.Tensor,
        event: torch.cuda.Event | None,
    ) -> None:
        nonlocal buffer_idx
        with use_stream(broadcast_streams, buffer_idx):
            if event is not None:
                torch.cuda.current_stream().wait_event(event)
            _, header_ref = broadcast_header(
                header,
                group=group,
                src=src,
                device=payload.device,
            )
            record_header_stream(header_ref)
            if payload.numel() > 0:
                group.broadcast(payload, src=src)
                record_tensor_stream(payload)
            live_refs[buffer_idx] = (header_ref, payload)
        buffer_idx = (buffer_idx + 1) % len(broadcast_streams)

    read_idx = 0
    encode_idx = 0
    pack_idx = 0

    def read_next_chunk() -> TensorBatch:
        nonlocal pending_item, read_idx
        with use_stream(encode_streams, read_idx):
            chunk, pending_item = next_chunk(
                tensor_iterator,
                chunk_size,
                pending_item=pending_item,
            )
        read_idx = (read_idx + 1) % len(encode_streams)
        return chunk

    def prepare_chunk_payload(
        chunk: TensorBatch,
    ) -> tuple[dict[str, Any], torch.Tensor, torch.cuda.Event | None]:
        nonlocal encode_idx, pack_idx
        with use_stream(encode_streams, encode_idx):
            pending_payloads.queue_chunk(chunk, delta_tracker)
        encode_idx = (encode_idx + 1) % len(encode_streams)
        with use_stream(encode_streams, pack_idx):
            header, payload = pending_payloads.pop()
            event = record_stream_event(encode_streams[pack_idx])
        pack_idx = (pack_idx + 1) % len(encode_streams)
        return header, payload, event

    try:
        chunk = read_next_chunk()
        while chunk:
            header, payload, event = prepare_chunk_payload(chunk)
            next_weight_chunk = read_next_chunk()
            send(header, payload, event)
            chunk = next_weight_chunk

        pending_payloads.flush_sparse_bucket()
        while pending_payloads:
            with use_stream(encode_streams, pack_idx):
                header, payload = pending_payloads.pop()
                event = record_stream_event(encode_streams[pack_idx])
            pack_idx = (pack_idx + 1) % len(encode_streams)
            send(header, payload, event)

        sync_streams(encode_streams)
        sync_streams(broadcast_streams)
        broadcast_header(
            {"kind": G_TRANSFER_DONE_KIND},
            group=group,
            src=src,
            device=transfer_device,
        )
        synchronize_current_transfer_stream(transfer_device)
    except Exception:
        if delta_tracker is not None:
            delta_tracker.on_sync_failed()
        raise

    if delta_tracker is not None and is_delta_sync:
        delta_tracker.on_sync_succeeded()


def _participate_in_source_iteration_and_drain(
    *,
    iterator: Iterator[NamedTensor],
    group: Any,
    src: int,
    device: torch.device | int | str,
) -> None:
    """Keep non-source policy ranks in exporter collectives while rank 0 sends.

    Megatron-Bridge's HF export iterator performs tensor/expert-parallel
    collectives. Every policy worker must consume that iterator even though only
    the source rank contributes payload bytes to the model-update group.
    """
    streams = cuda_streams(device)
    buffer_idx = 0
    live_refs: list[Any] = [None for _ in streams]
    chunk_size = target_chunk_size()
    pending_item = None
    read_idx = 0
    transfer_done = False

    def read_next_chunk() -> TensorBatch:
        nonlocal pending_item, read_idx
        with use_stream(streams, read_idx):
            chunk, pending_item = next_chunk(
                iterator,
                chunk_size,
                pending_item=pending_item,
            )
        read_idx = (read_idx + 1) % len(streams)
        return chunk

    def drain_one() -> bool:
        nonlocal buffer_idx
        with use_stream(streams, buffer_idx):
            header, header_ref = broadcast_header(
                {},
                group=group,
                src=src,
                device=device,
            )
            record_header_stream(header_ref)
            if header["kind"] == G_TRANSFER_DONE_KIND:
                live_refs[buffer_idx] = (header_ref, None)
                return True
            payload = recv_payload(
                int(header["payload_numel"]),
                group=group,
                src=src,
                device=device,
            )
            record_tensor_stream(payload)
            live_refs[buffer_idx] = (header_ref, payload)
        buffer_idx = (buffer_idx + 1) % len(streams)
        return False

    try:
        chunk = read_next_chunk()
        while chunk:
            next_weight_chunk = read_next_chunk()
            if drain_one():
                transfer_done = True
                break
            chunk = next_weight_chunk
        while not transfer_done:
            transfer_done = drain_one()
    finally:
        sync_streams(streams)
        synchronize_current_transfer_stream(device)


def packed_weight_transfer_consumer(
    *,
    group: Any,
    src: int,
    load_full_weights_func: WeightLoadFunc,
    load_sparse_weights_func: SparseWeightLoadFunc,
    device: torch.device | int | str,
) -> WeightTransferResult:
    """Receive dense or sparse-delta chunks from ``packed_weight_transfer_producer``."""
    streams = cuda_streams(device)
    buffer_idx = 0
    live_refs: list[Any] = [None for _ in streams]
    transfer_done = False
    loaded_any = False
    is_delta_sync = False

    try:
        while True:
            with use_stream(streams, buffer_idx):
                header, header_ref = broadcast_header(
                    {},
                    group=group,
                    src=src,
                    device=device,
                )
                record_header_stream(header_ref)
                if header["kind"] == G_TRANSFER_DONE_KIND:
                    transfer_done = True
                    live_refs[buffer_idx] = (header_ref, None)
                    break

                is_delta_sync = is_delta_sync or bool(header.get("is_delta_sync"))
                payload_numel = int(header["payload_numel"])
                payload_tensors: TensorBatch = []
                if payload_numel > 0:
                    payload = recv_payload(
                        payload_numel,
                        group=group,
                        src=src,
                        device=device,
                    )
                    record_tensor_stream(payload)
                    payload_tensors = unpack_named_tensors(
                        payload,
                        entries=header["payload_entries"],
                    )
                else:
                    payload = None
                live_refs[buffer_idx] = (header_ref, payload)

                if header["kind"] == G_FULL_UPDATE_KIND:
                    load_full_weights_func(payload_tensors)
                    loaded_any = loaded_any or bool(payload_tensors)
                elif header["kind"] == G_DELTA_UPDATE_KIND:
                    metadata = header.get("sparse_metadata", [])
                    if payload_tensors and metadata:
                        load_sparse_weights_func(payload_tensors, metadata)
                        loaded_any = True
                    elif metadata:
                        raise RuntimeError(
                            "Sparse refit metadata arrived without payload."
                        )
                else:
                    raise ValueError(
                        f"Unsupported weight transfer kind: {header['kind']!r}"
                    )

            buffer_idx = (buffer_idx + 1) % len(streams)
    finally:
        if transfer_done:
            sync_streams(streams)

    return WeightTransferResult(loaded_any=loaded_any, is_delta_sync=is_delta_sync)
