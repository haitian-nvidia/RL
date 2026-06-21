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

import io

import pytest
import torch

from nemo_rl.models.generation.vllm.config import VllmDeltaCompressionConfig
from nemo_rl.utils import weight_transfer_sparse_codec as sparse_codec
from nemo_rl.utils.weight_transfer import (
    packed_weight_transfer_consumer,
    packed_weight_transfer_producer,
)
from nemo_rl.utils.weight_transfer_delta_tracker import (
    DeltaCompressionTracker,
    create_vllm_delta_transfer_tracker,
)
from nemo_rl.utils.weight_transfer_http import (
    G_VLLM_REFIT_API_KEY_HEADER,
    decode_vllm_refit_request_body,
    encode_vllm_refit_request_body,
    stream_sparse_delta_payloads_via_http,
    vllm_refit_api_key_headers,
    vllm_refit_api_key_is_valid,
)
from nemo_rl.utils.weight_transfer_protocol import (
    G_DELTA_UPDATE_KIND,
    G_DENSE_TRANSPORT,
    G_INDEX_END_KEY,
    G_INDEX_START_KEY,
    G_PACKED_INDICES_NAME,
    G_PACKED_VALUES_NAME,
    G_SPARSE_INDICES_TRANSPORT,
    G_TRANSFER_DONE_KIND,
    additive_weight_load_context,
    broadcast_header,
    pack_named_tensors,
    unpack_named_tensors,
)


class _NoopGroup:
    rank = 0

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        del tensor, src


class _QueuedBroadcastGroup:
    def __init__(self, rank: int, queue: list[torch.Tensor]) -> None:
        self.rank = rank
        self._queue = queue

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        if self.rank == src:
            self._queue.append(tensor.detach().cpu().clone())
            return
        queued = self._queue.pop(0)
        tensor.copy_(queued.to(device=tensor.device, dtype=tensor.dtype))


def _tracker(
    full_sync_interval: int = 3,
    index_encoding: str = "indices",
) -> DeltaCompressionTracker:
    return DeltaCompressionTracker(
        {
            "full_sync_interval": full_sync_interval,
            "sparse_bucket_size_bytes": 1024,
            "dtype": "float32",
            "index_encoding": index_encoding,
        }
    )


def test_pack_named_tensors_round_trips_mixed_dtypes() -> None:
    tensors = [
        ("weight", torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)),
        ("bias", torch.arange(3, dtype=torch.int32)),
    ]

    packed, entries = pack_named_tensors(tensors)
    unpacked = unpack_named_tensors(packed, entries)

    assert [name for name, _ in unpacked] == ["weight", "bias"]
    for (_, expected), (_, actual) in zip(tensors, unpacked, strict=True):
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert torch.equal(actual, expected)


def test_additive_weight_load_context_handles_param_data_and_views() -> None:
    param = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3))
    untouched = torch.zeros(2, dtype=torch.float32)

    with additive_weight_load_context([param]):
        param.data.copy_(torch.ones_like(param))
        param.data.view(-1).narrow(0, 2, 2).copy_(torch.tensor([10.0, 20.0]))
        untouched.copy_(torch.ones_like(untouched))

    torch.testing.assert_close(
        param,
        torch.tensor([[1.0, 2.0, 13.0], [24.0, 5.0, 6.0]]),
    )
    torch.testing.assert_close(untouched, torch.ones_like(untouched))


def test_merge_sparse_payloads_offsets_metadata() -> None:
    payload_a = sparse_codec.encode_sparse_infos(
        [
            (
                "a",
                torch.zeros(4, dtype=torch.float32),
                torch.tensor([1, 3], dtype=torch.int64),
                torch.tensor([0.5, 0.75], dtype=torch.float32),
            )
        ],
    )
    payload_b = sparse_codec.encode_sparse_infos(
        [
            (
                "b",
                torch.zeros(3, dtype=torch.float32),
                torch.tensor([0], dtype=torch.int64),
                torch.tensor([2.0], dtype=torch.float32),
            )
        ],
    )

    tensors, _, metadata = sparse_codec.merge_sparse_payloads([payload_a, payload_b])
    packed = dict(tensors)

    assert torch.equal(
        packed[G_PACKED_INDICES_NAME], torch.tensor([1, 3], dtype=torch.int32)
    )
    assert torch.equal(packed[G_PACKED_VALUES_NAME], torch.tensor([0.5, 0.75, 2.0]))
    assert metadata[0][G_INDEX_START_KEY] == 0
    assert metadata[0][G_INDEX_END_KEY] == 2
    assert metadata[1][G_INDEX_START_KEY] == 2
    assert metadata[1][G_INDEX_END_KEY] == 2
    assert metadata[1]["value_start"] == 2
    assert metadata[1]["value_end"] == 3


def test_sparse_indices_keep_int32_for_small_flat_locations() -> None:
    locations = torch.tensor([1, 3], dtype=torch.int64)
    tensor = torch.zeros(8, dtype=torch.float32)
    payload_tensors, _transport, metadata = sparse_codec.encode_sparse_infos(
        [
            (
                "linear.weight",
                tensor,
                locations,
                torch.tensor([2.0, 4.0], dtype=torch.float32),
            )
        ],
        index_encoding="indices",
    )
    packed = dict(payload_tensors)[G_PACKED_INDICES_NAME]

    assert packed.dtype == torch.int32
    assert metadata[0]["index_encoding"] == "indices"
    assert metadata[0]["explicit_index_width"] == 4


def test_sparse_indices_use_int64_for_large_flat_locations() -> None:
    large_location = 2**31 + 5
    # Use a non-contiguous, multi-element location set so the explicit-indices
    # path is exercised; a single location collapses to a range encoding.
    locations = torch.tensor([0, large_location], dtype=torch.int64)
    tensor = torch.empty(0, dtype=torch.float32)
    payload_tensors, _transport, metadata = sparse_codec.encode_sparse_infos(
        [
            (
                "huge.weight",
                tensor,
                locations,
                torch.tensor([6.0, 7.0], dtype=torch.float32),
            )
        ],
        index_encoding="indices",
    )
    packed = dict(payload_tensors)[G_PACKED_INDICES_NAME]

    assert packed.dtype == torch.int64
    assert metadata[0]["explicit_index_width"] == 8
    decoded = sparse_codec.sparse_locations_for_item(
        metadata[0],
        packed,
        (0, 2, 0, 2),
        device="cpu",
    )
    assert decoded.tolist() == [0, large_location]


def test_merge_sparse_indices_promotes_mixed_int_widths() -> None:
    # Each payload needs a non-contiguous, multi-element location set so the
    # explicit-indices path is exercised; a single location collapses to a range.
    small_payload = sparse_codec.encode_sparse_infos(
        [
            (
                "small.weight",
                torch.zeros(16, dtype=torch.float32),
                torch.tensor([3, 9], dtype=torch.int64),
                torch.tensor([1.0, 1.5], dtype=torch.float32),
            )
        ],
        index_encoding="indices",
    )
    large_payload = sparse_codec.encode_sparse_infos(
        [
            (
                "large.weight",
                torch.empty(0, dtype=torch.float32),
                torch.tensor([0, 2**31 + 1], dtype=torch.int64),
                torch.tensor([2.0, 2.5], dtype=torch.float32),
            )
        ],
        index_encoding="indices",
    )

    tensors, _transport, metadata = sparse_codec.merge_sparse_payloads(
        [small_payload, large_payload]
    )
    packed = dict(tensors)[G_PACKED_INDICES_NAME]

    assert packed.dtype == torch.int64
    assert [item["explicit_index_width"] for item in metadata] == [8, 8]


def test_collective_consumer_decodes_non_source_sparse_delta_header(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    queue: list[torch.Tensor] = []
    source = _QueuedBroadcastGroup(rank=0, queue=queue)
    receiver = _QueuedBroadcastGroup(rank=1, queue=queue)
    payload_tensors, transport, metadata = sparse_codec.encode_sparse_infos(
        [
            (
                "linear.weight",
                torch.zeros(8, dtype=torch.float32),
                torch.tensor([1, 4], dtype=torch.int64),
                torch.tensor([2.0, 3.0], dtype=torch.float32),
            )
        ],
        index_encoding="deltas",
    )
    packed_payload, payload_entries = pack_named_tensors(payload_tensors)
    assert transport == G_SPARSE_INDICES_TRANSPORT
    header = {
        "kind": G_DELTA_UPDATE_KIND,
        "transport": transport,
        "payload_entries": payload_entries,
        "payload_numel": int(packed_payload.numel()),
        "sparse_metadata": metadata,
        "is_delta_sync": True,
    }

    broadcast_header(header, group=source, src=0, device="cpu")
    source.broadcast(packed_payload, src=0)
    broadcast_header({"kind": G_TRANSFER_DONE_KIND}, group=source, src=0, device="cpu")
    loaded_sparse: list[tuple[list[tuple[str, torch.Tensor]], list[dict]]] = []

    result = packed_weight_transfer_consumer(
        group=receiver,
        src=0,
        load_full_weights_func=lambda tensors: pytest.fail(
            f"unexpected full payload: {tensors}"
        ),
        load_sparse_weights_func=lambda tensors, sparse_metadata: loaded_sparse.append(
            (tensors, sparse_metadata)
        ),
        device="cpu",
    )

    assert result.loaded_any
    assert result.is_delta_sync
    assert not queue
    assert len(loaded_sparse) == 1
    loaded_tensors, loaded_metadata = loaded_sparse[0]
    assert loaded_metadata[0]["index_encoding"] == "deltas"
    assert loaded_metadata[0]["name"] == "linear.weight"
    decoded = dict(
        item
        for batch in sparse_codec.decode_sparse(
            loaded_tensors,
            loaded_metadata,
            device="cpu",
            byte_cap=1024,
        )
        for item in batch
    )
    torch.testing.assert_close(
        decoded["linear.weight"],
        torch.tensor([0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0]),
    )


def test_collective_full_sync_defers_source_baseline_prewarm(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("NRL_REFIT_CPU_TARGET_PACKED_TENSOR_SIZE", "1024")
    monkeypatch.setenv("NRL_REFIT_BASELINE_IN_MEMORY", "1")
    tracker = _tracker()
    tensor = torch.tensor([1.0, 2.0, 3.0])

    packed_weight_transfer_producer(
        [("linear.weight", tensor)],
        group=_NoopGroup(),
        src=0,
        delta_tracker=tracker,
    )

    assert tracker.committed_syncs == 0
    assert tracker.has_pending_full_sync_baseline()
    tracker.snapshot_pending_full_sync_baseline([("linear.weight", tensor)])
    tracker.on_sync_succeeded()
    assert tracker.committed_syncs == 1
    assert torch.equal(tracker.baseline["linear.weight"], tensor)


def test_collective_delta_sync_updates_source_baseline(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("NRL_REFIT_CPU_TARGET_PACKED_TENSOR_SIZE", "1024")
    monkeypatch.setenv("NRL_REFIT_BASELINE_IN_MEMORY", "1")
    tracker = _tracker()
    tensor = torch.tensor([1.0, 2.0, 3.0])
    tracker.prepare_sparse_delta_payload([("linear.weight", tensor)])
    tracker.snapshot_pending_full_sync_baseline([("linear.weight", tensor)])
    tracker.on_sync_succeeded()

    tensor.add_(torch.tensor([0.0, 4.0, 0.0]))
    packed_weight_transfer_producer(
        [("linear.weight", tensor)],
        group=_NoopGroup(),
        src=0,
        delta_tracker=tracker,
    )

    assert tracker.committed_syncs == 2
    assert torch.equal(tracker.baseline["linear.weight"], tensor)


def test_delta_tracker_accepts_pydantic_delta_config(monkeypatch) -> None:
    monkeypatch.setenv("NRL_REFIT_BASELINE_IN_MEMORY", "1")
    tracker = create_vllm_delta_transfer_tracker(
        {
            "delta_compression": VllmDeltaCompressionConfig(
                enabled=True,
                dtype="float32",
                full_sync_interval=3,
                sparse_bucket_size_bytes=1024,
                delta_load_batch_size_bytes=1024,
                index_encoding="deltas",
            )
        }
    )

    assert tracker is not None
    assert tracker.index_encoding == "deltas"


def test_delta_tracker_respects_periodic_full_sync_interval(monkeypatch) -> None:
    monkeypatch.setenv("NRL_REFIT_BASELINE_IN_MEMORY", "1")
    tracker = _tracker(full_sync_interval=3)
    tensor = torch.zeros(4, dtype=torch.float32)

    is_delta, payload = tracker.prepare_sparse_delta_payload(
        [("linear.weight", tensor)]
    )
    assert not is_delta
    tracker.snapshot_pending_full_sync_baseline(payload)
    tracker.on_sync_succeeded()
    assert tracker.is_delta_sync()

    for value in (1.0, 2.0):
        tensor[0] = value
        is_delta, _payload = tracker.prepare_sparse_delta_payload(
            [("linear.weight", tensor)]
        )
        assert is_delta
        tracker.on_sync_succeeded()

    assert tracker.committed_syncs == 3
    assert not tracker.is_delta_sync()
    tensor[0] = 3.0
    is_delta, payload = tracker.prepare_sparse_delta_payload(
        [("linear.weight", tensor)]
    )

    assert not is_delta
    assert len(payload) == 1
    assert payload[0][0] == "linear.weight"
    assert payload[0][1] is tensor
    assert tracker.has_pending_full_sync_baseline()


def test_http_periodic_full_sync_posts_dense_payload_and_flushes(
    monkeypatch,
) -> None:
    monkeypatch.setenv("NRL_REFIT_BASELINE_IN_MEMORY", "1")
    monkeypatch.setenv("NRL_REFIT_HTTP_INFLIGHT_BUCKETS", "1")
    monkeypatch.setattr(
        "nemo_rl.utils.weight_transfer_http.get_target_packed_tensor_size",
        lambda: 1024,
    )
    tracker = _tracker(full_sync_interval=2)
    tensor = torch.zeros(4, dtype=torch.float32)
    is_delta, payload = tracker.prepare_sparse_delta_payload(
        [("linear.weight", tensor)]
    )
    assert not is_delta
    tracker.snapshot_pending_full_sync_baseline(payload)
    tracker.on_sync_succeeded()

    tensor[0] = 1.0
    is_delta, _payload = tracker.prepare_sparse_delta_payload(
        [("linear.weight", tensor)]
    )
    assert is_delta
    tracker.on_sync_succeeded()
    assert not tracker.is_delta_sync()

    requests = []

    def fake_post(
        endpoint_urls,
        body,
        *,
        api_key_env_var,
        timeout_s,
        content_type="application/octet-stream",
        extra_headers=None,
    ):
        del endpoint_urls, api_key_env_var, timeout_s, content_type, extra_headers
        requests.append(torch.load(io.BytesIO(body), weights_only=True))

    flushes = []

    def fake_flush(base_urls, *, api_key_env_var, timeout_s):
        flushes.append((base_urls, api_key_env_var, timeout_s))
        return {"ok": True, "receiver_total_s": 0.25}

    monkeypatch.setattr(
        "nemo_rl.utils.weight_transfer_http._post_refit_body_to_endpoint_urls",
        fake_post,
    )
    monkeypatch.setattr(
        "nemo_rl.utils.weight_transfer_http.flush_vllm_refit_urls",
        fake_flush,
    )
    tensor[1] = 2.0

    result = stream_sparse_delta_payloads_via_http(
        [("linear.weight", tensor)],
        delta_tracker=tracker,
        is_payload_source=True,
        refit_urls=["http://worker"],
    )

    assert result["payloads"] == 1
    assert len(requests) == 1
    assert requests[0]["transport"] == G_DENSE_TRANSPORT
    assert requests[0]["metadata"] == []
    assert requests[0]["payload_tensors"][0][0] == "linear.weight"
    torch.testing.assert_close(requests[0]["payload_tensors"][0][1], tensor)
    assert flushes == [(["http://worker"], None, 600.0)]
    assert tracker.committed_syncs == 3
    torch.testing.assert_close(tracker.baseline["linear.weight"], tensor)


def test_http_refit_request_body_zlib_round_trips(monkeypatch) -> None:
    monkeypatch.setenv("NRL_REFIT_HTTP_BODY_COMPRESS", "zlib")
    body = (b"nemo-rl-refit-payload" * 1024) + torch.arange(32).numpy().tobytes()

    encoded, headers = encode_vllm_refit_request_body(body)

    assert headers["content-encoding"] == "zlib"
    assert int(headers["x-nemo-rl-refit-uncompressed-bytes"]) == len(body)
    assert len(encoded) < len(body)
    assert decode_vllm_refit_request_body(encoded, headers) == body


def test_vllm_refit_api_key_is_disabled_without_configured_env() -> None:
    assert vllm_refit_api_key_headers(None) == {}
    assert vllm_refit_api_key_is_valid(None, {})


def test_vllm_refit_api_key_fails_closed_when_configured_but_unset(
    monkeypatch,
) -> None:
    monkeypatch.delenv("NRL_TEST_REFIT_KEY", raising=False)

    with pytest.raises(RuntimeError, match="configured but unset or empty"):
        vllm_refit_api_key_headers("NRL_TEST_REFIT_KEY")

    assert not vllm_refit_api_key_is_valid("NRL_TEST_REFIT_KEY", {})


def test_vllm_refit_api_key_requires_exact_header(monkeypatch) -> None:
    monkeypatch.setenv("NRL_TEST_REFIT_KEY", "secret")

    assert vllm_refit_api_key_headers("NRL_TEST_REFIT_KEY") == {
        G_VLLM_REFIT_API_KEY_HEADER: "secret"
    }
    assert vllm_refit_api_key_is_valid(
        "NRL_TEST_REFIT_KEY",
        {G_VLLM_REFIT_API_KEY_HEADER: "secret"},
    )
    assert not vllm_refit_api_key_is_valid("NRL_TEST_REFIT_KEY", {})
    assert not vllm_refit_api_key_is_valid(
        "NRL_TEST_REFIT_KEY",
        {G_VLLM_REFIT_API_KEY_HEADER: "wrong"},
    )


def test_delta_tracker_records_sparse_locations_without_rescan(monkeypatch) -> None:
    monkeypatch.setenv("NRL_REFIT_BASELINE_IN_MEMORY", "1")
    tracker = _tracker(index_encoding="deltas")
    tensor = torch.zeros(8, dtype=torch.float32)
    bias = torch.zeros(6, dtype=torch.float32)
    tracker.prepare_sparse_delta_payload([("linear.weight", tensor), ("bias", bias)])
    tracker.snapshot_pending_full_sync_baseline(
        [("linear.weight", tensor), ("bias", bias)]
    )
    tracker.on_sync_succeeded()

    assert tracker.record_sparse_delta_locations(
        "linear.weight",
        locations=torch.tensor([0, 3, 7]),
        values=torch.tensor([1.0, 2.0, 3.0]),
    )
    assert tracker.record_sparse_delta_range(
        "bias",
        start=2,
        values=torch.tensor([4.0, 5.0]),
    )
    is_delta, payload = tracker.prepare_sparse_delta_payload(
        [("linear.weight", tensor), ("bias", bias)]
    )

    assert is_delta
    payload_tensors, _transport, metadata = payload
    assert metadata[0]["index_encoding"] == "deltas"
    assert metadata[1]["index_encoding"] == "range"
    assert dict(payload_tensors)[G_PACKED_INDICES_NAME].dtype == torch.uint8
    decoded = dict(
        item
        for batch in sparse_codec.decode_sparse(
            payload_tensors,
            metadata,
            device="cpu",
            byte_cap=1024,
        )
        for item in batch
    )
    torch.testing.assert_close(
        decoded["linear.weight"],
        torch.tensor([1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]),
    )
    torch.testing.assert_close(
        decoded["bias"],
        torch.tensor([0.0, 0.0, 4.0, 5.0, 0.0, 0.0]),
    )

    tracker.on_sync_succeeded()
    torch.testing.assert_close(
        tracker.baseline["linear.weight"],
        torch.tensor([1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]),
    )
    torch.testing.assert_close(
        tracker.baseline["bias"],
        torch.tensor([0.0, 0.0, 4.0, 5.0, 0.0, 0.0]),
    )


def test_collective_delta_sync_rejects_dense_fallback(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("NRL_REFIT_CPU_TARGET_PACKED_TENSOR_SIZE", "1024")
    tracker = _tracker()
    tracker.committed_syncs = 1

    with pytest.raises(RuntimeError, match="dense payload during a delta sync"):
        packed_weight_transfer_producer(
            [("missing_baseline.weight", torch.ones(2))],
            group=_NoopGroup(),
            src=0,
            delta_tracker=tracker,
        )


def test_delta_tracker_failed_full_sync_clears_pending_baseline(monkeypatch) -> None:
    monkeypatch.setenv("NRL_REFIT_BASELINE_IN_MEMORY", "1")
    tracker = _tracker()

    is_delta, payload = tracker.prepare_sparse_delta_payload(
        [("linear.weight", torch.ones(2))]
    )

    assert not is_delta
    assert payload
    assert tracker.has_pending_full_sync_baseline()
    tracker.on_sync_failed()
    assert not tracker.has_pending_full_sync_baseline()
