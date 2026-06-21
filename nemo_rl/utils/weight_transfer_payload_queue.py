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

from collections import deque
from typing import Any, cast

import torch

from nemo_rl.utils import weight_transfer_sparse_codec as sparse_codec
from nemo_rl.utils.weight_transfer_delta_tracker import DeltaCompressionTracker
from nemo_rl.utils.weight_transfer_protocol import (
    G_DELTA_UPDATE_KIND,
    G_DENSE_TRANSPORT,
    G_FULL_UPDATE_KIND,
    QueuedPayload,
    SparseBucketPayload,
    TensorBatch,
    TensorPayload,
    WeightTransferKind,
    pack_named_tensors,
    record_payload_readiness_events,
    wait_for_payload_events,
    wire_bytes,
)


class PendingWirePayloads:
    """Queue dense and sparse refit payloads before NCCL framing."""

    def __init__(
        self,
        *,
        transfer_device: torch.device,
        sparse_bucket_size_bytes: int,
        is_delta_sync: bool,
    ) -> None:
        # pyrefly mis-detects torch.device as a read-only descriptor on assignment.
        self._transfer_device = transfer_device  # pyrefly: ignore[read-only]
        self._sparse_bucket_size_bytes = sparse_bucket_size_bytes
        self._is_delta_sync = is_delta_sync
        self._queued: deque[QueuedPayload] = deque()
        self._sparse_bucket: list[SparseBucketPayload] = []
        self._sparse_bucket_bytes = 0

    def __bool__(self) -> bool:
        return bool(self._queued)

    def queue_payload(self, kind: WeightTransferKind, payload: TensorPayload) -> None:
        if kind != G_DELTA_UPDATE_KIND:
            self.flush_sparse_bucket()
            self._queued.append((kind, payload, record_payload_readiness_events()))
            return

        tensors, _, metadata = payload
        payload_bytes = wire_bytes(tensors)
        if (
            self._sparse_bucket
            and self._sparse_bucket_bytes + payload_bytes
            > self._sparse_bucket_size_bytes
        ):
            self.flush_sparse_bucket()

        if metadata:
            self._sparse_bucket.append((payload, record_payload_readiness_events()))
            self._sparse_bucket_bytes += payload_bytes

        if self._sparse_bucket_bytes >= self._sparse_bucket_size_bytes:
            self.flush_sparse_bucket()

    def queue_chunk(
        self,
        chunk: TensorBatch,
        delta_tracker: DeltaCompressionTracker | None,
    ) -> None:
        if delta_tracker is None:
            self.queue_payload(G_FULL_UPDATE_KIND, (chunk, G_DENSE_TRANSPORT, []))
            return

        delta_sync = delta_tracker.is_delta_sync()
        is_delta, payload_or_tensors = delta_tracker.prepare_sparse_delta_payload(
            chunk,
            target_device=self._transfer_device,
        )
        if not is_delta:
            if delta_sync:
                raise RuntimeError(
                    "Delta collective refit attempted to send a dense payload during "
                    "a delta sync. This indicates a missing or stale delta baseline."
                )
            self.queue_payload(
                G_FULL_UPDATE_KIND,
                (cast(TensorBatch, payload_or_tensors), G_DENSE_TRANSPORT, []),
            )
            return

        self.queue_payload(G_DELTA_UPDATE_KIND, cast(TensorPayload, payload_or_tensors))

    def flush_sparse_bucket(self) -> None:
        if not self._sparse_bucket:
            return

        payloads: list[TensorPayload] = [payload for payload, _ in self._sparse_bucket]
        wait_for_payload_events(
            event for _, events in self._sparse_bucket for event in events
        )
        self._queued.append(
            (
                G_DELTA_UPDATE_KIND,
                (
                    payloads[0]
                    if len(payloads) == 1
                    else sparse_codec.merge_sparse_payloads(payloads)
                ),
                record_payload_readiness_events(),
            )
        )
        self._sparse_bucket = []
        self._sparse_bucket_bytes = 0

    def pop(self) -> tuple[dict[str, Any], torch.Tensor]:
        kind, payload, ready_events = self._queued.popleft()
        wait_for_payload_events(ready_events)
        tensors, transport, metadata = payload
        header: dict[str, Any] = {
            "kind": kind,
            "transport": transport,
            "sparse_metadata": metadata,
            "is_delta_sync": self._is_delta_sync,
        }
        if not tensors:
            header.update({"payload_entries": [], "payload_numel": 0})
            return header, torch.empty(
                0,
                dtype=torch.uint8,
                device=self._transfer_device,
            )

        packed, entries = pack_named_tensors(tensors)
        header.update(
            {"payload_entries": entries, "payload_numel": int(packed.numel())}
        )
        return header, packed
