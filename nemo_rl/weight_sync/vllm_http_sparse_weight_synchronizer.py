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

"""Sparse HTTP weight synchronizer for remote non-colocated vLLM refit."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext, suppress
from typing import Any

import ray

from nemo_rl.models.generation.interfaces import GenerationInterface
from nemo_rl.utils.timer import Timer
from nemo_rl.utils.weight_transfer_http import (
    check_vllm_refit_health,
    normalize_vllm_refit_base_urls,
)
from nemo_rl.weight_sync.interfaces import WeightSynchronizer

_TRANSFER_TIMER_LABEL = "prepare_for_generation/transfer_and_update_weights"


class VllmHTTPSparseWeightSynchronizer(WeightSynchronizer):
    def __init__(
        self,
        policy: Any,
        generation: GenerationInterface | None,
        *,
        refit_urls: Sequence[str] | None = None,
        api_key_env_var: str | None = None,
        request_timeout_s: float = 600.0,
        health_timeout_s: float = 30.0,
        prepare_generation_refit_info: bool = True,
    ) -> None:
        self._policy = policy
        self._generation = generation
        self._configured_refit_urls = list(refit_urls or ())
        self._refit_urls: list[str] = []
        self._api_key_env_var = api_key_env_var
        self._request_timeout_s = request_timeout_s
        self._health_timeout_s = health_timeout_s
        self._stale = True
        self._initialized = False
        self._baseline_init_refs: list[Any] | None = None
        self._prepare_generation_refit_info = prepare_generation_refit_info

    def sync_weights(
        self,
        *,
        timer: Timer | None = None,
        kv_scales: dict[str, float] | None = None,
    ) -> None:
        if not self._initialized:
            self.init_communicator(kv_scales=kv_scales)

        timer_context = nullcontext()
        if timer is not None and not timer.is_running(_TRANSFER_TIMER_LABEL):
            timer_context = timer.time(_TRANSFER_TIMER_LABEL)
        with timer_context:
            if self._generation is not None:
                flush_success = self._generation.invalidate_kv_cache()
                if not flush_success:
                    print(
                        "vLLM KV cache invalidation failed before HTTP weight update."
                    )

            self.wait_for_baseline_initialization()
            futures = self._policy.stream_sparse_weights_via_http(
                self._refit_urls,
                api_key_env_var=self._api_key_env_var,
                timeout_s=self._request_timeout_s,
                kv_scales=kv_scales,
            )
            results = ray.get(futures)

        failed_results = [
            result
            for result in results
            if result is not None
            and (not isinstance(result, dict) or result.get("ok") is False)
        ]
        if failed_results:
            raise RuntimeError(
                f"vLLM HTTP sparse weight transfer failed: {failed_results}"
            )

        self._stale = False

    @property
    def is_stale(self) -> bool:
        return self._stale

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def mark_stale(self) -> None:
        self._stale = True

    def init_communicator(
        self,
        *,
        kv_scales: dict[str, float] | None = None,
    ) -> None:
        baseline_kwargs = {} if kv_scales is None else {"kv_scales": kv_scales}
        state_dict_info = None
        if self._prepare_generation_refit_info and self._generation is not None:
            state_dict_info = self._policy.prepare_refit_info()
        try:
            self._baseline_init_refs = self._policy.init_remote_sparse_delta_baseline(
                **baseline_kwargs
            )
            if state_dict_info is not None:
                self._generation.prepare_refit_info(state_dict_info)
            self._refit_urls = self._resolve_refit_urls()
            check_vllm_refit_health(
                self._refit_urls,
                api_key_env_var=self._api_key_env_var,
                timeout_s=self._health_timeout_s,
            )
            self._initialized = True
            self._stale = False
        except Exception:
            if self._baseline_init_refs is not None:
                for ref in self._baseline_init_refs:
                    with suppress(Exception):
                        ray.cancel(ref, force=True)
            self._baseline_init_refs = None
            self._refit_urls = []
            self._initialized = False
            self._stale = True
            raise

    def shutdown(self) -> None:
        try:
            self.wait_for_baseline_initialization()
        finally:
            self._baseline_init_refs = None
            self._refit_urls = []
            self._initialized = False
            self._stale = True

    def baseline_initialization_done(self) -> bool:
        if not self._baseline_init_refs:
            return True
        _, remaining = ray.wait(
            self._baseline_init_refs,
            num_returns=len(self._baseline_init_refs),
            timeout=0,
        )
        return not remaining

    def wait_for_baseline_initialization(self) -> None:
        if self._baseline_init_refs is None:
            return
        ray.get(self._baseline_init_refs)
        self._baseline_init_refs = None

    def _resolve_refit_urls(self) -> list[str]:
        urls = normalize_vllm_refit_base_urls(self._configured_refit_urls)
        if not urls and self._generation is not None:
            urls = normalize_vllm_refit_base_urls(
                self._generation.report_refit_server_base_urls()
            )
        if not urls:
            raise ValueError(
                "vLLM HTTP sparse refit requires at least one refit URL. "
                "Set refit_transport='vllm_http_sparse' with refit_urls, or "
                "enable vllm_cfg.expose_http_refit_server on generation workers."
            )
        return urls
