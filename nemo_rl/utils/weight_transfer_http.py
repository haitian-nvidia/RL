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

"""HTTP helpers for sparse vLLM refit payload transfer."""

from __future__ import annotations

import asyncio
import io
import json
import os
import threading
import time
import zlib
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from typing import Any, cast

import torch

from nemo_rl.distributed.virtual_cluster import _get_free_port_local, _get_node_ip_local
from nemo_rl.utils.weight_transfer_delta_tracker import DeltaCompressionTracker
from nemo_rl.utils.weight_transfer_protocol import (
    G_DENSE_TRANSPORT,
    G_SPARSE_INDICES_TRANSPORT,
    NamedTensor,
    TensorBatch,
    TensorPayload,
    env_int,
    get_target_packed_tensor_size,
    is_refit_receiver_timing,
    next_chunk,
)

G_VLLM_REFIT_SPARSE_DELTA_PATH = "/nemo-rl/refit/sparse-delta"
G_VLLM_REFIT_HEALTH_PATH = "/nemo-rl/refit/health"
G_VLLM_REFIT_FLUSH_PATH = "/nemo-rl/refit/flush"
G_VLLM_GENERATE_PATH = "/nemo-rl/generate"
G_VLLM_REFIT_API_KEY_HEADER = "x-nemo-rl-refit-key"
G_VLLM_REFIT_ASYNC_RECEIVER_APPLY_ENV = "NRL_REFIT_ASYNC_RECEIVER_APPLY"
G_VLLM_REFIT_HTTP_POOL_MAXSIZE_ENV = "NRL_REFIT_HTTP_POOL_MAXSIZE"
G_VLLM_REFIT_HTTP_POST_PARALLELISM_ENV = "NRL_REFIT_HTTP_POST_PARALLELISM"
G_VLLM_REFIT_HTTP_INFLIGHT_BUCKETS_ENV = "NRL_REFIT_HTTP_INFLIGHT_BUCKETS"
G_VLLM_REFIT_HTTP_BODY_COMPRESS_ENV = "NRL_REFIT_HTTP_BODY_COMPRESS"
G_VLLM_REFIT_HTTP_PROGRESS_INTERVAL_ENV = "NRL_REFIT_HTTP_PROGRESS_INTERVAL_S"
G_GENERATION_RESPONSE_KEYS = (
    "output_ids",
    "logprobs",
    "generation_lengths",
    "unpadded_sequence_lengths",
    "truncated",
)
G_DEFAULT_INFLIGHT_BUCKETS = 1
G_DEFAULT_PROGRESS_INTERVAL_S = 30

_HTTP_SESSION_LOCAL = threading.local()
_EXECUTORS: dict[tuple[str, int], ThreadPoolExecutor] = {}
_EXECUTORS_LOCK = threading.Lock()


def _get_keepalive_session() -> Any:
    session = getattr(_HTTP_SESSION_LOCAL, "session", None)
    if session is not None:
        return session
    try:
        import requests
    except ImportError:
        _HTTP_SESSION_LOCAL.session = False
        return False
    pool_size = env_int(G_VLLM_REFIT_HTTP_POOL_MAXSIZE_ENV, default=64, min_value=1)
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=pool_size,
        pool_maxsize=pool_size,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    _HTTP_SESSION_LOCAL.session = session
    return session


def normalize_vllm_refit_base_urls(refit_urls: Sequence[str]) -> list[str]:
    suffixes = (
        G_VLLM_REFIT_SPARSE_DELTA_PATH,
        G_VLLM_REFIT_HEALTH_PATH,
        G_VLLM_REFIT_FLUSH_PATH,
        G_VLLM_GENERATE_PATH,
    )
    urls = []
    for raw_url in refit_urls:
        url = raw_url.strip()
        for suffix in suffixes:
            if url.endswith(suffix):
                url = url[: -len(suffix)]
                break
        if url:
            urls.append(url.rstrip("/"))
    return urls


def vllm_refit_sparse_delta_url(base_url: str) -> str:
    return base_url.rstrip("/") + G_VLLM_REFIT_SPARSE_DELTA_PATH


def vllm_refit_health_url(base_url: str) -> str:
    return base_url.rstrip("/") + G_VLLM_REFIT_HEALTH_PATH


def vllm_refit_flush_url(base_url: str) -> str:
    return base_url.rstrip("/") + G_VLLM_REFIT_FLUSH_PATH


def vllm_generate_url(base_url: str) -> str:
    return base_url.rstrip("/") + G_VLLM_GENERATE_PATH


def vllm_refit_api_key_headers(api_key_env_var: str | None) -> dict[str, str]:
    if not api_key_env_var:
        return {}
    expected = os.environ.get(api_key_env_var)
    if not expected:
        raise RuntimeError(
            "vLLM HTTP refit API key env var "
            f"{api_key_env_var!r} is configured but unset or empty."
        )
    return {G_VLLM_REFIT_API_KEY_HEADER: expected}


def vllm_refit_api_key_is_valid(
    api_key_env_var: str | None,
    headers: Mapping[str, str],
) -> bool:
    if not api_key_env_var:
        return True
    expected = os.environ.get(api_key_env_var)
    return bool(expected) and headers.get(G_VLLM_REFIT_API_KEY_HEADER) == expected


def _http_post_parallelism(url_count: int) -> int:
    value = env_int(
        G_VLLM_REFIT_HTTP_POST_PARALLELISM_ENV,
        default=max(1, url_count),
        min_value=1,
    )
    return min(value, max(1, url_count))


def _inflight_bucket_window() -> int:
    return env_int(
        G_VLLM_REFIT_HTTP_INFLIGHT_BUCKETS_ENV,
        default=G_DEFAULT_INFLIGHT_BUCKETS,
        min_value=1,
    )


def _http_body_compression_mode() -> str | None:
    raw_mode = os.getenv(G_VLLM_REFIT_HTTP_BODY_COMPRESS_ENV, "")
    mode = raw_mode.strip().lower()
    if mode in {"", "0", "false", "none", "off"}:
        return None
    if mode in {"1", "true", "yes", "on"}:
        return "zstd"
    if mode not in {"zlib", "zstd"}:
        raise ValueError(
            f"Unsupported vLLM HTTP refit body compression {raw_mode!r}; "
            "expected zlib or zstd."
        )
    return mode


def encode_vllm_refit_request_body(body: bytes) -> tuple[bytes, dict[str, str]]:
    mode = _http_body_compression_mode()
    if mode is None or not body:
        return body, {}
    if mode == "zlib":
        encoded = zlib.compress(body, level=1)
    else:
        encoded = _zstd_compress(body)
    return (
        encoded,
        {
            "content-encoding": mode,
            "x-nemo-rl-refit-uncompressed-bytes": str(len(body)),
        },
    )


def decode_vllm_refit_request_body(
    body: bytes,
    headers: Mapping[str, str],
) -> bytes:
    encoding = headers.get("content-encoding") or headers.get("Content-Encoding")
    if encoding is None or encoding.strip().lower() in {"", "identity"}:
        return body
    mode = encoding.strip().lower()
    if mode == "zlib":
        return zlib.decompress(body)
    if mode == "zstd":
        return _zstd_decompress(body)
    raise ValueError(f"Unsupported vLLM HTTP refit request encoding {encoding!r}.")


def _executor(key: str, workers: int) -> ThreadPoolExecutor:
    with _EXECUTORS_LOCK:
        cache_key = (key, workers)
        if cache_key not in _EXECUTORS:
            _EXECUTORS[cache_key] = ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix=f"nrl-{key}",
            )
        return _EXECUTORS[cache_key]


def _map_parallel(
    key: str,
    items: Sequence[Any],
    fn: Callable[[Any], Any],
    parallelism: int,
) -> list[Any]:
    if parallelism == 1 or len(items) == 1:
        return [fn(item) for item in items]
    return list(_executor(key, parallelism).map(fn, items))


def _validate_generation_response(result: dict[str, Any], batch_size: int) -> None:
    if not result.get("ok", False):
        raise RuntimeError(f"vLLM HTTP generation shard failed: {result}")
    missing = [
        key
        for key in G_GENERATION_RESPONSE_KEYS
        if not isinstance(result.get(key), list) or len(result[key]) != batch_size
    ]
    if missing:
        raise RuntimeError(f"Incomplete vLLM HTTP generation response: {missing}")


def _generation_shard_body(
    payload: dict[str, Any],
    indices: list[int],
) -> bytes:
    shard = dict(payload)
    for key in ("input_ids", "input_lengths", "stop_strings"):
        if isinstance(payload.get(key), list):
            shard[key] = [payload[key][idx] for idx in indices]
    return json.dumps(shard).encode("utf-8")


def post_generation_payload_to_urls(
    generation_urls: Sequence[str],
    payload: dict[str, Any],
    *,
    api_key_env_var: str | None = None,
    timeout_s: float = 600.0,
) -> dict[str, Any]:
    urls = normalize_vllm_refit_base_urls(generation_urls)
    input_ids = payload.get("input_ids")
    input_lengths = payload.get("input_lengths")
    if not urls:
        raise ValueError("At least one vLLM generation URL is required")
    if not isinstance(input_ids, list) or not isinstance(input_lengths, list):
        raise TypeError("Generation payload requires list input_ids and input_lengths")
    if len(input_ids) != len(input_lengths):
        raise ValueError("Generation payload input_ids/input_lengths length mismatch")
    batch_size = len(input_ids)
    if batch_size == 0:
        return {"ok": True, **{key: [] for key in G_GENERATION_RESPONSE_KEYS}}
    if len(urls) == 1:
        result = _http_request_json(
            vllm_generate_url(urls[0]),
            json.dumps(payload).encode("utf-8"),
            api_key_env_var=api_key_env_var,
            timeout_s=timeout_s,
            content_type="application/json",
        )
        _validate_generation_response(result, batch_size)
        return {key: result[key] for key in ("ok", *G_GENERATION_RESPONSE_KEYS)}

    shard_count = min(len(urls), batch_size)
    shards = [
        (url, list(range(shard_idx, batch_size, shard_count)))
        for shard_idx, url in enumerate(urls[:shard_count])
    ]

    def post_one(url: str, indices: list[int]) -> tuple[list[int], dict[str, Any]]:
        return (
            indices,
            _http_request_json(
                vllm_generate_url(url),
                _generation_shard_body(payload, indices),
                api_key_env_var=api_key_env_var,
                timeout_s=timeout_s,
                content_type="application/json",
            ),
        )

    results = _map_parallel(
        "refit-gen",
        shards,
        lambda item: post_one(*item),
        _http_post_parallelism(len(shards)),
    )

    merged: dict[str, list[Any]] = {
        key: [None] * batch_size for key in G_GENERATION_RESPONSE_KEYS
    }
    for indices, result in results:
        _validate_generation_response(result, len(indices))
        for result_idx, original_idx in enumerate(indices):
            for key in merged:
                merged[key][original_idx] = result[key][result_idx]
    return {"ok": True, **merged}


def _serialize_refit_payload(payload: TensorPayload) -> bytes:
    payload_tensors, transport, metadata = payload
    if transport not in {G_DENSE_TRANSPORT, G_SPARSE_INDICES_TRANSPORT}:
        raise ValueError(f"vLLM HTTP refit got transport={transport!r}.")
    if transport == G_DENSE_TRANSPORT and metadata:
        raise ValueError("Dense vLLM HTTP refit payloads cannot carry metadata.")
    request = {
        "transport": transport,
        "metadata": list(metadata),
        "payload_tensors": [
            (name, tensor.detach().cpu()) for name, tensor in payload_tensors
        ],
    }
    buffer = io.BytesIO()
    torch.save(request, buffer)
    return buffer.getvalue()


def check_vllm_refit_health(
    refit_urls: Sequence[str],
    *,
    api_key_env_var: str | None = None,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    urls = [
        vllm_refit_health_url(url) for url in normalize_vllm_refit_base_urls(refit_urls)
    ]
    if not urls:
        raise ValueError("At least one vLLM HTTP refit URL is required.")
    responses = _post_refit_body_to_endpoint_urls(
        urls,
        None,
        api_key_env_var=api_key_env_var,
        timeout_s=timeout_s,
    )
    return {"ok": True, "urls": len(urls), "responses": responses}


def _drain_iterator(iterator: Iterable[NamedTensor]) -> None:
    for _ in iterator:
        pass


def init_sparse_delta_baseline_from_iterator(
    iterator: Iterable[NamedTensor],
    *,
    delta_tracker: DeltaCompressionTracker | None,
    is_payload_source: bool,
) -> dict[str, Any]:
    if delta_tracker is None:
        if is_payload_source:
            raise RuntimeError("vLLM HTTP sparse refit requires delta compression.")
        _drain_iterator(iterator)
        return {"payload_source": False, "baseline_initialized": False}
    if not is_payload_source:
        _drain_iterator(iterator)
        return {"payload_source": False, "baseline_initialized": False}
    if delta_tracker.full_sync_interval <= 1:
        raise ValueError("vLLM HTTP sparse refit requires full_sync_interval > 1.")

    tensor_iterator = iter(iterator)
    pending_item = None
    chunk_count = 0
    initialized = 0
    start_s = time.perf_counter()
    last_progress_s = start_s
    progress_interval_s = env_int(
        G_VLLM_REFIT_HTTP_PROGRESS_INTERVAL_ENV,
        default=G_DEFAULT_PROGRESS_INTERVAL_S,
        min_value=1,
    )
    print("REFIT_BASELINE_INIT event=start mode=remote_sparse_http", flush=True)
    while True:
        chunk, pending_item = next_chunk(
            tensor_iterator,
            get_target_packed_tensor_size(),
            pending_item=pending_item,
        )
        if not chunk:
            break
        chunk_count += 1
        is_delta, prepared = delta_tracker.prepare_sparse_delta_payload(
            chunk,
            target_device=None,
        )
        if not is_delta:
            dense_tensors = cast(TensorBatch, prepared)
            delta_tracker.snapshot_pending_full_sync_baseline(dense_tensors)
            initialized += len(dense_tensors)
        now_s = time.perf_counter()
        if now_s - last_progress_s >= progress_interval_s:
            print(
                "REFIT_BASELINE_INIT "
                f"event=progress chunks={chunk_count} tensors={initialized} "
                f"seconds={now_s - start_s:.3f}",
                flush=True,
            )
            last_progress_s = now_s
    delta_tracker.on_sync_succeeded()
    print(
        "REFIT_BASELINE_INIT "
        f"event=end chunks={chunk_count} tensors={initialized} "
        f"seconds={time.perf_counter() - start_s:.3f}",
        flush=True,
    )
    return {
        "payload_source": True,
        "baseline_initialized": True,
        "tensors": initialized,
    }


def stream_sparse_delta_payloads_via_http(
    iterator: Iterable[NamedTensor],
    *,
    delta_tracker: DeltaCompressionTracker | None,
    is_payload_source: bool,
    refit_urls: Sequence[str],
    api_key_env_var: str | None = None,
    timeout_s: float = 600.0,
) -> dict[str, Any]:
    urls = normalize_vllm_refit_base_urls(refit_urls)
    if not urls:
        raise ValueError("At least one vLLM HTTP refit URL is required.")
    if delta_tracker is None:
        if is_payload_source:
            raise RuntimeError("vLLM HTTP sparse refit requires delta compression.")
        _drain_iterator(iterator)
        return {"payload_source": False, "payloads": 0, "bytes": 0}
    if not is_payload_source:
        _drain_iterator(iterator)
        return {"payload_source": False, "payloads": 0, "bytes": 0}
    if delta_tracker.full_sync_interval <= 1:
        raise ValueError("vLLM HTTP sparse refit requires full_sync_interval > 1.")

    endpoint_urls = [vllm_refit_sparse_delta_url(url) for url in urls]
    window = _inflight_bucket_window()
    executor = _executor("refit-post", window)
    inflight: deque[Any] = deque()
    tensor_iterator = iter(iterator)
    pending_item = None
    payload_count = 0
    dense_payload_count = 0
    posted_bytes = 0
    export_pull_s = encode_s = d2h_s = post_wait_s = post_busy_s = flush_wait_s = 0.0
    stream_start = time.perf_counter()
    last_progress_s = stream_start
    progress_interval_s = env_int(
        G_VLLM_REFIT_HTTP_PROGRESS_INTERVAL_ENV,
        default=G_DEFAULT_PROGRESS_INTERVAL_S,
        min_value=1,
    )
    chunk_count = 0

    def post_payload(payload: TensorPayload) -> tuple[int, float]:
        started = time.perf_counter()
        body = _serialize_refit_payload(payload)
        body, extra_headers = encode_vllm_refit_request_body(body)
        _post_refit_body_to_endpoint_urls(
            endpoint_urls,
            body,
            api_key_env_var=api_key_env_var,
            timeout_s=timeout_s,
            extra_headers=extra_headers,
        )
        return len(body), time.perf_counter() - started

    def drain_one() -> None:
        nonlocal payload_count, posted_bytes, post_wait_s, post_busy_s
        wait_started = time.perf_counter()
        future = _pop_completed_or_oldest_future(inflight)
        nbytes, busy_s = future.result()
        post_wait_s = post_wait_s + (time.perf_counter() - wait_started)
        post_busy_s = post_busy_s + busy_s
        posted_bytes = posted_bytes + nbytes * len(urls)
        payload_count = payload_count + 1

    def submit_payload(payload: TensorPayload) -> None:
        nonlocal d2h_s
        started = time.perf_counter()
        while len(inflight) >= window:
            drain_one()
        inflight.append(executor.submit(post_payload, payload))
        d2h_s = d2h_s + (time.perf_counter() - started)

    def maybe_log_progress() -> None:
        nonlocal last_progress_s
        now_s = time.perf_counter()
        if now_s - last_progress_s < progress_interval_s:
            return
        print(
            "REFIT_HTTP_PROGRESS "
            f"chunks={chunk_count} "
            f"submitted_payloads={payload_count + len(inflight)} "
            f"completed_payloads={payload_count} "
            f"posted_mb={posted_bytes / 1e6:.1f} "
            f"seconds={now_s - stream_start:.3f}",
            flush=True,
        )
        last_progress_s = now_s

    try:
        while True:
            started = time.perf_counter()
            chunk, pending_item = next_chunk(
                tensor_iterator,
                get_target_packed_tensor_size(),
                pending_item=pending_item,
            )
            export_pull_s += time.perf_counter() - started
            if not chunk:
                break
            chunk_count += 1
            started = time.perf_counter()
            is_delta, prepared = delta_tracker.prepare_sparse_delta_payload(
                chunk,
                target_device=None,
            )
            encode_s += time.perf_counter() - started
            if not is_delta:
                dense_tensors = cast(TensorBatch, prepared)
                delta_tracker.snapshot_pending_full_sync_baseline(dense_tensors)
                payload: TensorPayload = (dense_tensors, G_DENSE_TRANSPORT, [])
                dense_payload_count += 1
            else:
                payload = cast(TensorPayload, prepared)
            _, transport, metadata = payload
            if transport == G_DENSE_TRANSPORT:
                submit_payload(payload)
                maybe_log_progress()
                continue
            if transport != G_SPARSE_INDICES_TRANSPORT:
                raise RuntimeError(
                    f"Unsupported vLLM HTTP refit transport: {transport!r}"
                )
            if not metadata:
                continue
            submit_payload(payload)
            maybe_log_progress()
        while inflight:
            drain_one()
        flush_result: dict[str, Any] = {}
        if payload_count and (
            dense_payload_count or delta_tracker.async_receiver_apply
        ):
            started = time.perf_counter()
            flush_result = flush_vllm_refit_urls(
                urls,
                api_key_env_var=api_key_env_var,
                timeout_s=timeout_s,
            )
            flush_wait_s = time.perf_counter() - started
        delta_tracker.on_sync_succeeded()
    except Exception:
        for future in inflight:
            with suppress(Exception):
                future.result()
        delta_tracker.on_sync_failed()
        raise

    print(
        "REFIT_HTTP_TIMING "
        f"total_s={time.perf_counter() - stream_start:.3f} "
        f"export_pull_s={export_pull_s:.3f} encode_s={encode_s:.3f} "
        f"d2h_s={d2h_s:.3f} post_wait_s={post_wait_s:.3f} "
        f"post_busy_s={post_busy_s:.3f} flush_wait_s={flush_wait_s:.3f} "
        f"receiver_total_s={flush_result.get('receiver_total_s', 0.0):.3f} "
        f"payloads={payload_count} posted_mb={posted_bytes / 1e6:.1f} "
        f"window={window}",
        flush=True,
    )
    return {
        "payload_source": True,
        "payloads": payload_count,
        "bytes": posted_bytes,
        "urls": len(urls),
    }


def _pop_completed_or_oldest_future(inflight: deque[Any]) -> Any:
    for future in inflight:
        if future.done():
            inflight.remove(future)
            return future
    return inflight.popleft()


def post_sparse_delta_payload_to_urls(
    base_urls: Sequence[str],
    body: bytes,
    *,
    api_key_env_var: str | None,
    timeout_s: float,
) -> dict[str, Any]:
    endpoint_urls = [
        vllm_refit_sparse_delta_url(url)
        for url in normalize_vllm_refit_base_urls(base_urls)
    ]
    if not endpoint_urls:
        raise ValueError("At least one vLLM HTTP refit URL is required.")
    body, extra_headers = encode_vllm_refit_request_body(body)
    _post_refit_body_to_endpoint_urls(
        endpoint_urls,
        body,
        api_key_env_var=api_key_env_var,
        timeout_s=timeout_s,
        extra_headers=extra_headers,
    )
    return {
        "ok": True,
        "urls": len(endpoint_urls),
        "bytes": len(body) * len(endpoint_urls),
    }


def _post_refit_body_to_endpoint_urls(
    endpoint_urls: Sequence[str],
    body: bytes | None,
    *,
    api_key_env_var: str | None,
    timeout_s: float,
    content_type: str = "application/octet-stream",
    extra_headers: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    return _fanout_refit_requests(
        endpoint_urls,
        _http_post_parallelism(len(endpoint_urls)),
        lambda url: _http_request_json(
            url,
            body,
            api_key_env_var=api_key_env_var,
            timeout_s=timeout_s,
            content_type=content_type,
            extra_headers=extra_headers,
        ),
    )


def flush_vllm_refit_urls(
    base_urls: Sequence[str],
    *,
    api_key_env_var: str | None,
    timeout_s: float,
) -> dict[str, Any]:
    endpoint_urls = [
        vllm_refit_flush_url(url) for url in normalize_vllm_refit_base_urls(base_urls)
    ]
    responses = _post_refit_body_to_endpoint_urls(
        endpoint_urls,
        b"{}",
        api_key_env_var=api_key_env_var,
        timeout_s=timeout_s,
        content_type="application/json",
    )
    result = {"ok": True, "urls": len(endpoint_urls)}
    result.update(_aggregate_fanout_refit_responses(responses))
    return result


def _aggregate_fanout_refit_responses(
    responses: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    payloads = 0
    for response in responses:
        if isinstance(response.get("payloads"), int):
            payloads += response["payloads"]
        for key, value in response.items():
            if (
                (key == "seconds" and isinstance(value, (int, float)))
                or is_refit_receiver_timing(key, value)
            ) and not isinstance(value, bool):
                result[key] = max(result.get(key, 0.0), float(value))
    if payloads:
        result["payloads"] = payloads
    return result


def _fanout_refit_requests(
    endpoint_urls: Sequence[str],
    parallelism: int,
    request_one: Callable[[str], dict[str, Any]],
) -> list[dict[str, Any]]:
    if not endpoint_urls:
        raise ValueError("At least one vLLM HTTP refit URL is required.")
    responses = _map_parallel(
        "refit-fanout",
        endpoint_urls,
        request_one,
        parallelism,
    )
    for url, response in zip(endpoint_urls, responses, strict=False):
        _raise_for_refit_response(url, response)
    return responses


def start_vllm_refit_relay_server(
    base_urls: Sequence[str],
    *,
    host: str = "0.0.0.0",
    port: int | None = None,
    api_key_env_var: str | None = None,
    timeout_s: float = 600.0,
    advertised_host: str | None = None,
) -> tuple[threading.Thread, str, Any]:
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from starlette.requests import ClientDisconnect

    urls = normalize_vllm_refit_base_urls(base_urls)
    if not urls:
        raise ValueError("At least one local vLLM refit URL is required.")
    port = _get_free_port_local() if port is None else int(port)
    app = FastAPI()

    def token_is_valid(raw_request: Request) -> bool:
        return vllm_refit_api_key_is_valid(api_key_env_var, raw_request.headers)

    def error_response(error: str, status_code: int) -> JSONResponse:
        return JSONResponse(
            {"ok": False, "error": error, "urls": len(urls)}, status_code
        )

    async def relay_health(raw_request) -> JSONResponse:
        if not token_is_valid(raw_request):
            return error_response("unauthorized", 403)
        try:
            await asyncio.to_thread(
                check_vllm_refit_health,
                urls,
                api_key_env_var=api_key_env_var,
                timeout_s=min(timeout_s, 30.0),
            )
        except Exception as exc:
            return error_response(str(exc), 503)
        return JSONResponse({"ok": True, "urls": len(urls)})

    async def relay_sparse_delta(raw_request) -> JSONResponse:
        if not token_is_valid(raw_request):
            return error_response("unauthorized", 403)
        try:
            body = await raw_request.body()
        except ClientDisconnect:
            return error_response(
                "Client disconnected while uploading sparse payload", 499
            )
        try:
            body = decode_vllm_refit_request_body(body, raw_request.headers)
        except ValueError as exc:
            return error_response(str(exc), 400)
        try:
            result = await asyncio.to_thread(
                post_sparse_delta_payload_to_urls,
                urls,
                body,
                api_key_env_var=api_key_env_var,
                timeout_s=timeout_s,
            )
        except Exception as exc:
            return error_response(str(exc), 500)
        return JSONResponse(result)

    async def relay_flush(raw_request) -> JSONResponse:
        if not token_is_valid(raw_request):
            return error_response("unauthorized", 403)
        try:
            result = await asyncio.to_thread(
                flush_vllm_refit_urls,
                urls,
                api_key_env_var=api_key_env_var,
                timeout_s=timeout_s,
            )
        except Exception as exc:
            return error_response(str(exc), 500)
        return JSONResponse(result)

    async def relay_generate(raw_request) -> JSONResponse:
        if not token_is_valid(raw_request):
            return error_response("unauthorized", 403)
        try:
            payload = json.loads((await raw_request.body()).decode("utf-8"))
            result = await asyncio.to_thread(
                post_generation_payload_to_urls,
                urls,
                payload,
                api_key_env_var=api_key_env_var,
                timeout_s=timeout_s,
            )
        except Exception as exc:
            return error_response(str(exc), 500)
        return JSONResponse(result)

    for handler in (relay_health, relay_sparse_delta, relay_flush, relay_generate):
        handler.__annotations__["raw_request"] = Request
    app.add_api_route(G_VLLM_REFIT_HEALTH_PATH, relay_health, methods=["GET"])
    app.add_api_route(
        G_VLLM_REFIT_SPARSE_DELTA_PATH, relay_sparse_delta, methods=["POST"]
    )
    app.add_api_route(G_VLLM_REFIT_FLUSH_PATH, relay_flush, methods=["POST"])
    app.add_api_route(G_VLLM_GENERATE_PATH, relay_generate, methods=["POST"])

    server = uvicorn.Server(
        uvicorn.Config(app, host=host, port=port, timeout_keep_alive=120)
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    relay_host = advertised_host or _get_node_ip_local()
    return thread, f"http://{relay_host}:{port}", server


def _raise_for_refit_response(url: str, response: dict[str, Any]) -> None:
    if not response.get("ok", False):
        raise RuntimeError(f"vLLM HTTP sparse refit failed for {url}: {response}")


def _http_request_json(
    url: str,
    body: bytes | None,
    *,
    api_key_env_var: str | None,
    timeout_s: float,
    content_type: str = "application/octet-stream",
    extra_headers: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    headers = {"content-type": content_type}
    if extra_headers:
        headers.update(extra_headers)
    headers.update(vllm_refit_api_key_headers(api_key_env_var))
    session = _get_keepalive_session()
    if session is False:
        raise RuntimeError("The requests package is required for vLLM HTTP refit.")
    response = (
        session.get(url, headers=headers, timeout=timeout_s)
        if body is None
        else session.post(url, data=body, headers=headers, timeout=timeout_s)
    )
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code} from {url}: {response.text}")
    return {} if not response.text else json.loads(response.text)


def _require_zstandard():
    try:
        import zstandard
    except ImportError:
        raise RuntimeError(
            "vLLM HTTP refit body compression 'zstd' requires the zstandard package."
        ) from None
    return zstandard


def _zstd_compress(raw: bytes) -> bytes:
    compressor = getattr(_HTTP_SESSION_LOCAL, "zstd_compressor", None)
    if compressor is None:
        compressor = _require_zstandard().ZstdCompressor(level=1)
        _HTTP_SESSION_LOCAL.zstd_compressor = compressor
    return compressor.compress(raw)


def _zstd_decompress(raw: bytes) -> bytes:
    decompressor = getattr(_HTTP_SESSION_LOCAL, "zstd_decompressor", None)
    if decompressor is None:
        decompressor = _require_zstandard().ZstdDecompressor()
        _HTTP_SESSION_LOCAL.zstd_decompressor = decompressor
    return decompressor.decompress(raw)
