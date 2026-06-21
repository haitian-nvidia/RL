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

"""External vLLM generation over NeMo-RL token-id HTTP endpoints."""

from __future__ import annotations

from typing import Any, NotRequired

import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation.interfaces import (
    GenerationConfig,
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.utils.weight_transfer_http import (
    normalize_vllm_refit_base_urls,
    post_generation_payload_to_urls,
)


class VllmHttpGenerationConfig(GenerationConfig):
    urls: list[str]
    refit_urls: NotRequired[list[str]]
    api_key_env_var: NotRequired[str | None]
    request_timeout_s: float
    refit_request_timeout_s: NotRequired[float]


class VllmHttpGeneration(GenerationInterface):
    def __init__(
        self,
        cluster: RayVirtualCluster | None,
        config: VllmHttpGenerationConfig,
    ) -> None:
        del cluster
        self.cfg = config
        self._generation_urls = normalize_vllm_refit_base_urls(config["urls"])
        self._refit_urls = normalize_vllm_refit_base_urls(
            config.get("refit_urls", self._generation_urls)
        )
        if not self._generation_urls:
            raise ValueError("vllm_http generation requires at least one URL")
        if not self._refit_urls:
            raise ValueError("vllm_http sparse refit requires at least one URL")
        self._api_key_env_var = config.get("api_key_env_var")
        self._request_timeout_s = float(config["request_timeout_s"])
        refit_request_timeout_s = config.get("refit_request_timeout_s")
        self._refit_request_timeout_s = (
            self._request_timeout_s
            if refit_request_timeout_s is None
            else float(refit_request_timeout_s)
        )

    def init_collective(self, ip: str, port: int, world_size: int) -> list[Any]:
        del ip, port, world_size
        return []

    def generate(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        greedy: bool = False,
    ) -> BatchedDataDict[GenerationOutputSpec]:
        verify_right_padding(data, pad_value=self.cfg["_pad_token_id"])
        payload: dict[str, Any] = {
            "input_ids": data["input_ids"].detach().cpu().tolist(),
            "input_lengths": data["input_lengths"].detach().cpu().tolist(),
            "greedy": greedy,
        }
        if "stop_strings" in data:
            payload["stop_strings"] = data["stop_strings"]

        result = post_generation_payload_to_urls(
            self._generation_urls,
            payload,
            api_key_env_var=self._api_key_env_var,
            timeout_s=self._request_timeout_s,
        )
        if not result.get("ok", False):
            raise RuntimeError(f"vLLM HTTP generation failed: {result}")

        return BatchedDataDict[GenerationOutputSpec](
            {
                "output_ids": _pad_rows(
                    result["output_ids"],
                    pad_value=int(self.cfg["_pad_token_id"]),
                    dtype=torch.long,
                ),
                "logprobs": _pad_rows(
                    result["logprobs"], pad_value=0.0, dtype=torch.float32
                ),
                "generation_lengths": torch.tensor(
                    result["generation_lengths"], dtype=torch.long
                ),
                "unpadded_sequence_lengths": torch.tensor(
                    result["unpadded_sequence_lengths"], dtype=torch.long
                ),
                "truncated": torch.tensor(result["truncated"], dtype=torch.bool),
            }
        )

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        del args, kwargs
        return True

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        del args, kwargs
        return True

    def shutdown(self) -> bool:
        return True

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        del state_dict_info

    def invalidate_kv_cache(self) -> bool:
        return True

    def report_refit_server_base_urls(self) -> list[str]:
        return list(self._refit_urls)

    @property
    def refit_api_key_env_var(self) -> str | None:
        return self._api_key_env_var

    @property
    def refit_request_timeout_s(self) -> float:
        return self._refit_request_timeout_s


def _pad_rows(
    rows: list[list[Any]],
    *,
    pad_value: Any,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not rows:
        return torch.empty((0, 0), dtype=dtype)
    width = max(len(row) for row in rows)
    if all(len(row) == width for row in rows):
        return torch.tensor(rows, dtype=dtype)
    tensor = torch.full((len(rows), width), pad_value, dtype=dtype)
    for row_idx, row in enumerate(rows):
        if row:
            tensor[row_idx, : len(row)] = torch.tensor(row, dtype=dtype)
    return tensor
