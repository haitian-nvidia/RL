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
"""Refitted Policy Comparison Script.

This script compares logprobs between a Megatron policy and a vLLM policy
after performing model weight refitting. It demonstrates the workflow for
getting consistent logprobs across different inference backends.

Usage:
    uv run --extra mcore python3 tools/refit_verifier.py --model_name /path/to/model


Example Output:

--- Comparing Logprobs ---

Input prompt: The following are multiple choice questions (with answers) about world religions.

When was the first Buddhist temple constructed in Japan?
A. 325 CE
B. 119 CE
C. 451 CE
D. 596 CE
Answer:
Input tokens: tensor([200000,    954,   2182,    583,   6146,   9031,   5808,    330,   5992,
      8860,     21,   1509,   3817,  99867,   1574,   7022,    812,    290,
      1660, 120819,  55594,  24043,    310,  11197,   1044,     45,     26,
       220,  23325,  13607,    198,     46,     26,    220,  12860,  13607,
       198,     47,     26,    220,  34518,  13607,    198,     48,     26,
       220,  43145,  13607,    198,   4984,     38])

Comparing 10 generated tokens (from position 51 to 60):
vLLM generated logprobs: tensor([-7.0227, -7.1559, -6.4603, -6.7419, -6.3026, -6.8391, -6.3128, -6.6454,
    -7.1514, -6.8304])
Megatron generated logprobs: tensor([-7.0225, -7.1873, -6.4600, -6.7418, -6.3027, -6.8704, -6.2502, -6.6453,
    -7.1518, -6.8304])
Absolute difference: tensor([2.0981e-04, 3.1348e-02, 2.6035e-04, 1.6689e-04, 1.4973e-04, 3.1272e-02,
    6.2590e-02, 1.7643e-04, 3.2902e-04, 4.1485e-05])
Mean absolute difference: 0.012654399499297142
Max absolute difference: 0.06259012222290039

--- Token-by-Token Comparison (Generated Tokens Only) ---
Token           Token ID   Position   vLLM         Megatron     Diff
---------------------------------------------------------------------------
tok_51          pos_51     51         -7.022674    -7.022464    0.000210
tok_52          pos_52     52         -7.155923    -7.187271    0.031348
tok_53          pos_53     53         -6.460307    -6.460047    0.000260
tok_54          pos_54     54         -6.741926    -6.741759    0.000167
tok_55          pos_55     55         -6.302569    -6.302719    0.000150
tok_56          pos_56     56         -6.839099    -6.870371    0.031272
tok_57          pos_57     57         -6.312774    -6.250184    0.062590
tok_58          pos_58     58         -6.645445    -6.645269    0.000176
tok_59          pos_59     59         -7.151441    -7.151770    0.000329
tok_60          pos_60     60         -6.830355    -6.830397    0.000041
"""

import argparse
import copy
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import ray
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from nemo_rl.algorithms.grpo import refit_policy_generation
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.constants import VLLM_BACKEND, VLLM_HTTP_BACKEND
from nemo_rl.models.generation.sglang.sglang_generation import SGLangGeneration
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.models.generation.vllm_http import VllmHttpGeneration
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.weight_transfer_http import start_vllm_refit_relay_server
from nemo_rl.weight_sync import create_weight_synchronizer


def _comma_separated_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare Megatron and vLLM policy logprobs after refitting"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="/root/checkpoints/llama4-scout-custom-init",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=1,
        help="Tensor parallelism size (TP) for Megatron",
    )
    parser.add_argument(
        "--ep_size",
        type=int,
        default=1,
        help="Expert parallelism size (EP) for Megatron",
    )
    parser.add_argument(
        "--pp_size",
        type=int,
        default=1,
        help="Pipeline parallelism size (PP) for Megatron",
    )
    for name in (
        "--expert_tensor_parallel_size",
        "--vllm_tp_size",
        "--vllm_ep_size",
        "--vllm_pp_size",
    ):
        parser.add_argument(name, type=int, default=None)
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=10,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=256,
        help="Maximum total sequence length",
    )
    parser.add_argument(
        "--refit_buffer_size_gb", type=int, default=4, help="Refit buffer size in GB"
    )
    parser.add_argument("--non_colocated", action="store_true")
    parser.add_argument(
        "--refit_transport",
        choices=("collective", "vllm_http_sparse"),
        default="collective",
    )
    parser.add_argument("--vllm_http_refit_urls", type=str, default="")
    for name in (
        "--external_vllm_http_refit",
        "--serve_vllm_http_refit",
        "--serve_vllm_http_refit_relay",
        "--enable_delta_compression",
        "--skip_logprob_compare",
    ):
        parser.add_argument(name, action="store_true")
    parser.add_argument("--serve_seconds", type=float, default=0.0)
    parser.add_argument("--vllm_http_refit_api_key_env_var", type=str, default=None)
    parser.add_argument("--vllm_http_refit_server_port", type=int, default=None)
    parser.add_argument("--vllm_http_refit_relay_port", type=int, default=None)
    parser.add_argument("--vllm_http_refit_timeout_s", type=float, default=600.0)
    parser.add_argument(
        "--overlap_initial_http_baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For external vLLM HTTP refit, run receiver generation while the "
            "source-side sparse-delta baseline is being built."
        ),
    )
    parser.add_argument(
        "--baseline_overlap_generation_batch_size",
        type=int,
        default=8,
        help="Batch size for generation used to overlap initial HTTP baseline setup.",
    )
    parser.add_argument(
        "--baseline_overlap_generation_timeout_s",
        type=float,
        default=0.0,
        help="Optional wall-clock cap for overlap generation; 0 means wait until done.",
    )
    parser.add_argument("--policy_num_nodes", type=int, default=1)
    parser.add_argument("--generation_num_nodes", type=int, default=1)
    parser.add_argument("--policy_gpus_per_node", type=int, default=None)
    parser.add_argument("--generation_gpus_per_node", type=int, default=None)
    parser.add_argument(
        "--master_port_range_low",
        type=int,
        default=None,
        help="Lower bound, inclusive, for RayVirtualCluster TCPStore master ports.",
    )
    parser.add_argument(
        "--master_port_range_high",
        type=int,
        default=None,
        help="Upper bound, exclusive, for RayVirtualCluster TCPStore master ports.",
    )
    parser.add_argument("--delta_full_sync_interval", type=int, default=20)
    parser.add_argument(
        "--delta_sparse_bucket_size_bytes", type=int, default=512 * 1024**2
    )
    parser.add_argument(
        "--delta_load_batch_size_bytes", type=int, default=512 * 1024**2
    )
    parser.add_argument(
        "--delta_index_encoding",
        choices=("indices", "deltas", "deltas_zstd"),
        default="indices",
    )
    parser.add_argument("--num_refits", type=int, default=1)
    parser.add_argument("--benchmark_sparse_update_fraction", type=float, default=0.0)
    parser.add_argument("--benchmark_sparse_update_delta", type=float, default=0.01)
    parser.add_argument("--benchmark_sparse_update_seed", type=int, default=1234)
    parser.add_argument(
        "--benchmark_sparse_update_pattern",
        choices=("contiguous", "strided"),
        default="contiguous",
    )
    parser.add_argument("--benchmark_label", type=str, default="refit")
    parser.add_argument(
        "--vllm_enforce_eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run vLLM in eager mode.",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.6,
        help="GPU memory utilization passed to vLLM.",
    )
    parser.add_argument(
        "--megatron_attention_backend",
        choices=("auto", "flash", "fused", "unfused", "local"),
        default=None,
        help="Optional Megatron attention backend override.",
    )
    parser.add_argument(
        "--sequence_parallel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Megatron sequence parallelism.",
    )
    parser.add_argument(
        "--moe_token_dispatcher_type",
        choices=("allgather", "alltoall", "flex"),
        default="allgather",
        help="Megatron MoE token dispatcher type.",
    )
    parser.add_argument(
        "--moe_flex_dispatcher_backend",
        choices=("deepep", "hybridep"),
        default=None,
        help="Optional Megatron flex dispatcher backend.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Here is a short introduction to me:",
        help="Input prompt for generation",
    )

    args = parser.parse_args()
    if (args.master_port_range_low is None) != (args.master_port_range_high is None):
        parser.error(
            "--master_port_range_low and --master_port_range_high must be set together"
        )
    if (
        args.master_port_range_low is not None
        and args.master_port_range_low >= args.master_port_range_high
    ):
        parser.error(
            "--master_port_range_low must be less than --master_port_range_high"
        )
    return args


def resolve_cluster_sizes(args) -> None:
    if args.expert_tensor_parallel_size is None:
        args.expert_tensor_parallel_size = args.tp_size
    if args.vllm_tp_size is None:
        args.vllm_tp_size = args.tp_size
    if args.vllm_ep_size is None:
        args.vllm_ep_size = args.ep_size
    if args.vllm_pp_size is None:
        args.vllm_pp_size = args.pp_size
    if args.policy_gpus_per_node is None:
        args.policy_gpus_per_node = args.tp_size * args.ep_size * args.pp_size
    if args.generation_gpus_per_node is None:
        args.generation_gpus_per_node = (
            max(args.vllm_tp_size, args.vllm_ep_size) * args.vllm_pp_size
        )


def master_port_range_kwargs(args) -> dict[str, int | None]:
    return {
        "port_range_low": args.master_port_range_low,
        "port_range_high": args.master_port_range_high,
    }


def setup_configs(args, tokenizer):
    """Setup configuration dictionaries for Megatron and vLLM.

    Args:
        args: Parsed command line arguments
        tokenizer: HuggingFace tokenizer

    Returns:
        tuple: (megatron_config, vllm_config)
    """
    colocated = not args.non_colocated
    tensor_pipeline_size = args.tp_size * args.pp_size
    policy_world_size = args.policy_num_nodes * args.policy_gpus_per_node
    if policy_world_size % tensor_pipeline_size != 0:
        raise ValueError(
            "Policy world size must be divisible by TP * PP: "
            f"policy_world_size={policy_world_size}, TP={args.tp_size}, "
            f"PP={args.pp_size}"
        )
    train_global_batch_size = policy_world_size // tensor_pipeline_size

    megatron_config = {
        "model_name": args.model_name,
        "training_backend": "megatron",
        "train_global_batch_size": train_global_batch_size,
        "train_micro_batch_size": 1,
        "generation_batch_size": 2,
        "learning_rate": 0.0001,
        "logprob_batch_size": 1,
        "generation": {
            "backend": VLLM_BACKEND,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": None,
            "max_total_sequence_length": args.max_sequence_length,
            "max_new_tokens": args.max_sequence_length,
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
            "colocated": {
                "enabled": colocated,
                "resources": {
                    "gpus_per_node": args.generation_gpus_per_node,
                    "num_nodes": args.generation_num_nodes,
                },
            },
        },
        "precision": "bfloat16",
        "offload_optimizer_for_logprob": False,
        "pipeline_dtype": "bfloat16",
        "parallel_output": True,
        "max_total_sequence_length": args.max_sequence_length,
        "fsdp_offload_enabled": False,
        "max_grad_norm": 1.0,
        "refit_buffer_size_gb": args.refit_buffer_size_gb,
        "make_sequence_length_divisible_by": args.tp_size,
        "optimizer": {
            "type": "adam",
            "kwargs": {
                "lr": 0.0001,
                "weight_decay": 0.0,
                "eps": 1e-8,
            },
        },
        "dtensor_cfg": {
            "enabled": False,
        },
        "dynamic_batching": {
            "enabled": False,
            "train_mb_tokens": 256,
            "logprob_mb_tokens": 256,
            "sequence_length_round": 64,
        },
        "sequence_packing": {
            "enabled": False,
        },
        "megatron_cfg": {
            "enabled": True,
            "empty_unused_memory_level": 1,
            "wrap_with_ddp": False,
            "tensor_model_parallel_size": args.tp_size,
            "sequence_parallel": args.sequence_parallel,
            "expert_tensor_parallel_size": args.expert_tensor_parallel_size,
            "expert_model_parallel_size": args.ep_size,
            "pipeline_model_parallel_size": args.pp_size,
            "context_parallel_size": 1,
            "num_layers_in_first_pipeline_stage": None,
            "num_layers_in_last_pipeline_stage": None,
            "activation_checkpointing": False,
            "moe_router_dtype": "fp64",
            "moe_router_load_balancing_type": "none",
            "moe_router_bias_update_rate": 0.0,
            "moe_permute_fusion": False,
            "moe_enable_deepep": False,
            "moe_token_dispatcher_type": args.moe_token_dispatcher_type,
            "moe_shared_expert_overlap": False,
            "moe_grouped_gemm": True,
            "pipeline_dtype": "bfloat16",
            "train_iters": 1,
            "bias_activation_fusion": False,
            "moe_per_layer_logging": False,
            "freeze_moe_router": False,
            "apply_rope_fusion": False,
            "gradient_accumulation_fusion": False,
            "optimizer": {
                "optimizer": "adam",
                "lr": 5.0e-6,
                "min_lr": 5.0e-7,
                "weight_decay": 0.01,
                "bf16": True,
                "fp16": False,
                "params_dtype": "float32",
                # Adam optimizer settings
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_eps": 1e-8,
                # SGD optimizer settings
                "sgd_momentum": 0.9,
                # Distributed optimizer settings
                "use_distributed_optimizer": True,
                "use_precision_aware_optimizer": True,
                "clip_grad": 1.0,
                # Optimizer CPU offload settings
                "optimizer_cpu_offload": False,
                "optimizer_offload_fraction": 0.0,
            },
            "scheduler": {
                "start_weight_decay": 0.01,
                "end_weight_decay": 0.01,
                "weight_decay_incr_style": "constant",
                "lr_decay_style": "constant",
                "lr_decay_iters": None,
                "lr_warmup_iters": 50,
                "lr_warmup_init": 5.0e-7,
            },
            "distributed_data_parallel_config": {
                "grad_reduce_in_fp32": False,
                "overlap_grad_reduce": False,
                "overlap_param_gather": False,
                "use_custom_fsdp": False,
                "data_parallel_sharding_strategy": "optim_grads_params",
            },
        },
    }
    if args.megatron_attention_backend is not None:
        megatron_config["megatron_cfg"]["attention_backend"] = (
            args.megatron_attention_backend
        )
    if args.moe_flex_dispatcher_backend is not None:
        megatron_config["megatron_cfg"]["moe_flex_dispatcher_backend"] = (
            args.moe_flex_dispatcher_backend
        )

    vllm_config = {
        "backend": VLLM_BACKEND,
        "model_name": args.model_name,
        "tokenizer": {
            "name": args.model_name,
        },
        "dtype": "bfloat16",
        "max_new_tokens": args.max_new_tokens,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "stop_token_ids": None,
        "stop_strings": None,
        "vllm_cfg": {
            "tensor_parallel_size": args.vllm_tp_size,
            "pipeline_parallel_size": args.vllm_pp_size,
            "expert_parallel_size": args.vllm_ep_size,
            "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            "max_model_len": args.max_sequence_length,
            "precision": "bfloat16",
            "async_engine": False,
            "skip_tokenizer_init": False,
            "load_format": (
                "auto" if args.refit_transport == "vllm_http_sparse" else "dummy"
            ),
            "enforce_eager": args.vllm_enforce_eager,
        },
        "colocated": {
            "enabled": colocated,
            "resources": {
                "gpus_per_node": args.generation_gpus_per_node,
                "num_nodes": args.generation_num_nodes,
            },
        },
        "vllm_kwargs": {},
    }
    if args.refit_transport == "vllm_http_sparse":
        vllm_config["vllm_cfg"]["expose_http_refit_server"] = True
        if args.vllm_http_refit_api_key_env_var:
            vllm_config["vllm_cfg"]["http_refit_api_key_env_var"] = (
                args.vllm_http_refit_api_key_env_var
            )
        if args.vllm_http_refit_server_port is not None:
            vllm_config["vllm_cfg"]["http_refit_server_port"] = (
                args.vllm_http_refit_server_port
            )
    if args.enable_delta_compression:
        if colocated:
            raise ValueError(
                "--enable_delta_compression requires --non_colocated because "
                "delta compression is only supported for non-colocated refit."
            )
        delta_compression_config = {
            "enabled": True,
            "dtype": "bfloat16",
            "full_sync_interval": args.delta_full_sync_interval,
            "sparse_bucket_size_bytes": args.delta_sparse_bucket_size_bytes,
            "delta_load_batch_size_bytes": args.delta_load_batch_size_bytes,
            "index_encoding": args.delta_index_encoding,
        }
        vllm_config["delta_compression"] = delta_compression_config
        megatron_config["generation"]["delta_compression"] = delta_compression_config
        megatron_config["generation"]["vllm_cfg"] = {
            "precision": vllm_config["vllm_cfg"]["precision"]
        }

    vllm_config = configure_generation_config(vllm_config, tokenizer)
    if args.refit_transport == "vllm_http_sparse":
        vllm_config["vllm_cfg"]["load_format"] = "auto"

    return megatron_config, vllm_config


def setup_clusters_and_policies(args, megatron_config, vllm_config, tokenizer):
    """Setup Ray clusters and initialize policies.

    Args:
        args: Parsed command line arguments
        megatron_config: Megatron configuration dictionary
        vllm_config: vLLM configuration dictionary
        tokenizer: HuggingFace tokenizer

    Returns:
        tuple: (megatron_cluster, generation_cluster, policy, vllm_inference_policy)
    """
    print(
        f"Setting up Megatron Cluster with {args.policy_num_nodes} node(s), "
        f"{args.policy_gpus_per_node} GPU(s)/node"
    )
    megatron_cluster = RayVirtualCluster(
        name="megatron_cluster",
        bundle_ct_per_node_list=[args.policy_gpus_per_node] * args.policy_num_nodes,
        use_gpus=True,
        num_gpus_per_node=args.policy_gpus_per_node,
        max_colocated_worker_groups=1 if args.non_colocated else 2,
        **master_port_range_kwargs(args),
    )

    print("Instantiating Policy with Megatron backend...")
    policy = Policy(
        cluster=megatron_cluster,
        config=megatron_config,
        tokenizer=tokenizer,
        init_reference_model=False,
        init_optimizer=False,
    )

    if args.external_vllm_http_refit:
        print(
            "Using external vLLM HTTP sparse refit endpoints; "
            "local generation cluster will not be created."
        )
        return megatron_cluster, None, policy, None

    generation_cluster = megatron_cluster
    if args.non_colocated:
        print(
            f"Setting up vLLM Cluster with {args.generation_num_nodes} node(s), "
            f"{args.generation_gpus_per_node} GPU(s)/node"
        )
        generation_cluster = RayVirtualCluster(
            name="vllm_cluster",
            bundle_ct_per_node_list=[args.generation_gpus_per_node]
            * args.generation_num_nodes,
            use_gpus=True,
            num_gpus_per_node=args.generation_gpus_per_node,
            max_colocated_worker_groups=1,
            placement_group_strategy="PACK",
            **master_port_range_kwargs(args),
        )

    vllm_inference_config = vllm_config.copy()
    vllm_inference_config["max_new_tokens"] = args.max_new_tokens
    vllm_inference_config = configure_generation_config(
        vllm_inference_config, tokenizer
    )
    vllm_inference_policy = VllmGeneration(
        cluster=generation_cluster,
        config=vllm_inference_config,
    )

    return megatron_cluster, generation_cluster, policy, vllm_inference_policy


def setup_vllm_http_refit_server(args, vllm_config):
    print(
        f"Setting up external vLLM refit server cluster with "
        f"{args.generation_num_nodes} node(s), "
        f"{args.generation_gpus_per_node} GPU(s)/node"
    )
    generation_cluster = RayVirtualCluster(
        name="vllm_refit_server_cluster",
        bundle_ct_per_node_list=[args.generation_gpus_per_node]
        * args.generation_num_nodes,
        use_gpus=True,
        num_gpus_per_node=args.generation_gpus_per_node,
        max_colocated_worker_groups=1,
        placement_group_strategy="PACK",
        **master_port_range_kwargs(args),
    )
    vllm_inference_policy = VllmGeneration(
        cluster=generation_cluster,
        config=vllm_config,
    )
    urls = vllm_inference_policy.report_refit_server_base_urls()
    print(f"REFIT_HTTP_SERVER_URLS count={len(urls)}", flush=True)
    for url in urls:
        print(f"REFIT_HTTP_SERVER_URL url={url}", flush=True)

    if args.serve_vllm_http_refit_relay:
        _, relay_url, _ = start_vllm_refit_relay_server(
            urls,
            port=args.vllm_http_refit_relay_port,
            api_key_env_var=args.vllm_http_refit_api_key_env_var,
            timeout_s=args.vllm_http_refit_timeout_s,
        )
        print("REFIT_HTTP_RELAY enabled=true", flush=True)
        print(f"REFIT_HTTP_RELAY_URL url={relay_url}", flush=True)

    return generation_cluster, vllm_inference_policy


def prepare_input_data(prompt, tokenizer):
    """Tokenize the input prompt and prepare generation data.

    Args:
        prompt: Input text prompt
        tokenizer: HuggingFace tokenizer

    Returns:
        BatchedDataDict: Prepared input data
    """
    print("Preparing input data...")

    # Tokenize the prompt
    tokenized = tokenizer(
        [prompt],
        padding=True,
        truncation=True,
        return_tensors="pt",
        padding_side="right",
    )

    # Calculate input lengths from attention mask
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    input_lengths = attention_mask.sum(dim=1).to(torch.int32)

    generation_data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
        }
    )

    return generation_data


def run_model_refitting(
    policy,
    vllm_inference_policy,
    weight_sync,
    args,
    *,
    source_baseline_only_full_sync=False,
    first_pass_is_delta=False,
):
    """Perform model weight refitting between Megatron and vLLM policies.

    Args:
        policy: Megatron policy
        vllm_inference_policy: vLLM inference policy
        weight_sync: Weight synchronizer for the policy/generation pair
        num_refits: Number of refit passes to run
    """
    print("\n--- Performing Model Refitting ---")
    if args.num_refits < 1:
        raise ValueError("--num_refits must be >= 1")
    if first_pass_is_delta:
        print(
            "REFIT_BENCHMARK_CONTROL_NOTE "
            "transport=vllm_http_sparse initial_baseline=overlapped "
            "reason=receiver_already_loaded_initial_checkpoint",
            flush=True,
        )
    elif source_baseline_only_full_sync:
        print(
            "REFIT_BENCHMARK_CONTROL_NOTE "
            "transport=vllm_http_sparse full_mode=source_baseline_only "
            "reason=receiver_already_loaded_initial_checkpoint",
            flush=True,
        )

    durations_by_mode: dict[str, list[float]] = {}
    for refit_idx in range(args.num_refits):
        refit_pass = refit_idx + 1
        is_delta_pass = args.enable_delta_compression and (
            refit_idx != 0 or first_pass_is_delta
        )
        mode = "delta" if is_delta_pass else "full"
        if not is_delta_pass and source_baseline_only_full_sync:
            mode = "source_baseline"

        print(f"Refit pass {refit_pass}/{args.num_refits}")
        started = time.perf_counter()
        if weight_sync is None:
            refit_policy_generation(
                policy,
                vllm_inference_policy,
                colocated_inference=True,
                _refit_buffer_size_gb=policy.cfg["refit_buffer_size_gb"],
            )
        else:
            weight_sync.sync_weights()
        elapsed = time.perf_counter() - started
        durations_by_mode.setdefault(mode, []).append(elapsed)
        print(
            "REFIT_BENCHMARK "
            f"label={args.benchmark_label} pass={refit_pass} "
            f"mode={mode} encoding={args.delta_index_encoding} seconds={elapsed:.6f}",
            flush=True,
        )
        if args.benchmark_sparse_update_fraction > 0 and refit_pass < args.num_refits:
            apply_sparse_update_for_benchmark(
                policy,
                pass_after=refit_pass,
                fraction=args.benchmark_sparse_update_fraction,
                delta=args.benchmark_sparse_update_delta,
                seed=args.benchmark_sparse_update_seed,
                pattern=args.benchmark_sparse_update_pattern,
            )

    for mode, durations in durations_by_mode.items():
        mean_duration = sum(durations) / len(durations)
        print(
            "REFIT_BENCHMARK_SUMMARY "
            f"label={args.benchmark_label} mode={mode} "
            f"encoding={args.delta_index_encoding} "
            f"count={len(durations)} "
            f"mean_seconds={mean_duration:.6f} "
            f"min_seconds={min(durations):.6f} "
            f"max_seconds={max(durations):.6f}",
            flush=True,
        )
    print("Model refitting completed")


def apply_sparse_update_for_benchmark(
    policy,
    *,
    pass_after,
    fraction,
    delta,
    seed,
    pattern,
):
    if fraction <= 0:
        return
    if fraction > 1:
        raise ValueError("--benchmark_sparse_update_fraction must be <= 1")

    started = time.perf_counter()
    results = policy.run_all_workers_single_data(
        "apply_refit_benchmark_sparse_update",
        fraction=fraction,
        delta=delta,
        seed=seed + pass_after,
        pattern=pattern,
    )
    elapsed = time.perf_counter() - started
    values = sum(int(result["values"]) for result in results)
    total_values = sum(int(result["total_values"]) for result in results)
    tensors = sum(int(result["tensors"]) for result in results)
    actual_fraction = values / total_values if total_values else 0.0
    stride = int(results[0]["stride"]) if results else 0
    pattern = str(results[0].get("pattern", "unknown")) if results else "unknown"
    print(
        "REFIT_BENCHMARK_UPDATE "
        f"pass_after={pass_after} seconds={elapsed:.6f} "
        f"requested_fraction={fraction:.6f} "
        f"actual_fraction={actual_fraction:.8f} "
        f"pattern={pattern} stride={stride} tensors={tensors} values={values} "
        f"total_values={total_values}",
        flush=True,
    )


def maybe_overlap_initial_http_baseline(
    policy,
    weight_sync,
    refit_urls: list[str],
    args,
    generation_data: BatchedDataDict,
    tokenizer,
) -> bool:
    if not args.external_vllm_http_refit or not args.overlap_initial_http_baseline:
        return False
    if not hasattr(weight_sync, "baseline_initialization_done"):
        return False

    batch_size = max(1, int(args.baseline_overlap_generation_batch_size))
    generation_policy = VllmHttpGeneration(
        cluster=None,
        config={
            "backend": VLLM_HTTP_BACKEND,
            "urls": refit_urls,
            "refit_urls": refit_urls,
            "api_key_env_var": args.vllm_http_refit_api_key_env_var,
            "request_timeout_s": args.vllm_http_refit_timeout_s,
            "refit_request_timeout_s": args.vllm_http_refit_timeout_s,
            "_pad_token_id": tokenizer.pad_token_id,
        },
    )
    overlap_data = _repeat_generation_data(generation_data, batch_size)
    started = time.perf_counter()
    iterations = 0
    generated_tokens = 0
    overlap_timeout = False
    print(
        "REFIT_BASELINE_OVERLAP event=start "
        f"batch_size={batch_size} timeout_s={args.baseline_overlap_generation_timeout_s}",
        flush=True,
    )
    while not weight_sync.baseline_initialization_done():
        if (
            args.baseline_overlap_generation_timeout_s > 0
            and time.perf_counter() - started
            >= args.baseline_overlap_generation_timeout_s
        ):
            overlap_timeout = True
            break
        result = generation_policy.generate(overlap_data, greedy=True)
        iterations += 1
        generated_tokens += int(result["generation_lengths"].sum().item())
        print(
            "REFIT_BASELINE_OVERLAP "
            f"event=progress iterations={iterations} "
            f"generated_tokens={generated_tokens} "
            f"seconds={time.perf_counter() - started:.3f}",
            flush=True,
        )

    weight_sync.wait_for_baseline_initialization()
    elapsed = time.perf_counter() - started
    print(
        "REFIT_BASELINE_OVERLAP "
        f"event=end iterations={iterations} generated_tokens={generated_tokens} "
        f"seconds={elapsed:.3f} timed_out={int(overlap_timeout)}",
        flush=True,
    )
    if args.benchmark_sparse_update_fraction > 0:
        apply_sparse_update_for_benchmark(
            policy,
            pass_after=0,
            fraction=args.benchmark_sparse_update_fraction,
            delta=args.benchmark_sparse_update_delta,
            seed=args.benchmark_sparse_update_seed,
            pattern=args.benchmark_sparse_update_pattern,
        )
    return True


def _repeat_generation_data(
    generation_data: BatchedDataDict,
    batch_size: int,
) -> BatchedDataDict:
    if batch_size <= 1:
        return generation_data
    data = BatchedDataDict(
        {
            "input_ids": generation_data["input_ids"].repeat(batch_size, 1),
            "input_lengths": generation_data["input_lengths"].repeat(batch_size),
        }
    )
    if "stop_strings" in generation_data:
        data["stop_strings"] = generation_data["stop_strings"] * batch_size
    return data


def generate_and_compare_logprobs(policy, vllm_inference_policy, generation_data):
    """Generate outputs and compare logprobs between vLLM and Megatron policies.

    Args:
        policy: Megatron policy
        vllm_inference_policy: vLLM inference policy
        generation_data: Input data for generation

    Returns:
        tuple: (vllm_logprobs_data, megatron_generation_data)
    """
    # Generate with vLLM for logprobs
    print("\n--- Getting vLLM Policy Logprobs ---")
    vllm_logprobs_data = vllm_inference_policy.generate(generation_data, greedy=True)
    print(f"vLLM Logprobs shape: {vllm_logprobs_data['logprobs'].shape}")
    print(f"vLLM Logprobs sample: {vllm_logprobs_data['logprobs'][0, -10:]}")

    # Generate with Megatron policy
    print("\n--- Getting Megatron Generation ---")
    policy.prepare_for_generation()

    # Prepare input data for Megatron using vLLM outputs
    megatron_input_data = copy.deepcopy(generation_data)
    print("=" * 100)
    print(megatron_input_data)
    print(vllm_logprobs_data)
    megatron_input_data["input_ids"] = vllm_logprobs_data["output_ids"]
    megatron_input_data["input_lengths"] = vllm_logprobs_data[
        "unpadded_sequence_lengths"
    ]

    # Get logprobs from Megatron
    policy.prepare_for_lp_inference()
    megatron_generation_data = policy.get_logprobs(megatron_input_data)
    print(f"Megatron Generation shape: {megatron_generation_data['logprobs'].shape}")
    print(
        f"Megatron Generation sample: {megatron_generation_data['logprobs'][0, -10:]}"
    )

    return vllm_logprobs_data, megatron_generation_data


def analyze_logprob_differences(
    vllm_logprobs_data, megatron_generation_data, generation_data, tokenizer, prompt
):
    """Analyze and display differences between vLLM and Megatron logprobs.

    Args:
        vllm_logprobs_data: vLLM generation results
        megatron_generation_data: Megatron generation results
        generation_data: Original input data
        tokenizer: HuggingFace tokenizer
        prompt: Original input prompt
    """
    print("\n--- Comparing Logprobs ---")
    print(f"Input prompt: {prompt}")
    print(
        f"Input tokens: {generation_data['input_ids'][0, : generation_data['input_lengths'][0]]}"
    )

    # Extract generation parameters
    input_length = generation_data["input_lengths"][0].item()
    total_length = vllm_logprobs_data["logprobs"].shape[1]
    generated_length = vllm_logprobs_data["generation_lengths"][0].item()

    if generated_length > 0:
        print(
            f"\nComparing {generated_length} generated tokens (from position {input_length} to {total_length - 1}):"
        )

        # Extract generated logprobs
        vllm_gen_logprobs = vllm_logprobs_data["logprobs"][0, input_length:total_length]
        megatron_gen_logprobs = megatron_generation_data["logprobs"][
            0, input_length:total_length
        ]

        print(f"vLLM generated logprobs: {vllm_gen_logprobs}")
        print(f"Megatron generated logprobs: {megatron_gen_logprobs}")

        # Calculate and display differences
        abs_diff = torch.abs(vllm_gen_logprobs - megatron_gen_logprobs)
        print(f"Absolute difference: {abs_diff}")
        print(f"Mean absolute difference: {torch.mean(abs_diff)}")
        print(f"Max absolute difference: {torch.max(abs_diff)}")

        # Detailed token-by-token comparison
        _detailed_token_comparison(
            vllm_gen_logprobs,
            megatron_gen_logprobs,
            vllm_logprobs_data,
            input_length,
            total_length,
            tokenizer,
        )
    else:
        print(
            f"No generated tokens to compare (input_length: {input_length}, total_length: {total_length})"
        )


def _detailed_token_comparison(
    vllm_logprobs,
    megatron_logprobs,
    vllm_logprobs_data,
    input_length,
    total_length,
    tokenizer,
):
    """Display detailed token-by-token comparison of logprobs.

    Args:
        vllm_logprobs: vLLM logprobs for generated tokens
        megatron_logprobs: Megatron logprobs for generated tokens
        vllm_logprobs_data: Vllm generation data
        input_length: Length of input sequence
        total_length: Total sequence length
        tokenizer: HuggingFace tokenizer
    """
    print("\n--- Token-by-Token Comparison (Generated Tokens Only) ---")

    if total_length > input_length:
        # Get generated tokens if available
        if "output_ids" in vllm_logprobs_data:
            generated_tokens = vllm_logprobs_data["output_ids"][
                0, input_length:total_length
            ]
        else:
            generated_tokens = torch.arange(input_length, total_length)

        # Display header
        print(
            f"{'Token':<15} {'Token ID':<10} {'Position':<10} {'vLLM':<12} {'Megatron':<12} {'Diff':<12}"
        )
        print("-" * 75)

        # Display each token comparison
        for i, pos in enumerate(range(input_length, total_length)):
            if "output_ids" in vllm_logprobs_data:
                token_id = generated_tokens[i].item()
                token_text = tokenizer.decode([token_id])
            else:
                token_id = f"pos_{pos}"
                token_text = f"tok_{pos}"

            vllm_lp = vllm_logprobs[i].item()
            megatron_lp = megatron_logprobs[i].item()
            diff = abs(vllm_lp - megatron_lp)

            print(
                f"{token_text:<15} {token_id:<10} {pos:<10} {vllm_lp:<12.6f} {megatron_lp:<12.6f} {diff:<12.6f}"
            )
    else:
        print("No generated tokens to compare in detail.")


def cleanup_resources(
    policy=None,
    vllm_inference_policy=None,
    megatron_cluster=None,
    generation_cluster=None,
):
    """Clean up resources and shutdown policies.

    Args:
        policy: Megatron policy to shutdown
        vllm_inference_policy: vLLM policy to shutdown
        megatron_cluster: Policy cluster to shutdown
        generation_cluster: Optional separate generation cluster to shutdown
    """
    print("\n--- Cleaning up ---")
    resources = [vllm_inference_policy, policy]
    if generation_cluster is not None and generation_cluster is not megatron_cluster:
        resources.append(generation_cluster)
    resources.append(megatron_cluster)

    for resource in resources:
        if resource is not None:
            resource.shutdown()
    print("Cleanup completed successfully!")


def init_sglang(inference_cluster, generation_config):
    """Initialize SGLang generation workers and snapshot/reset weights for verification.

    Args:
        inference_cluster: Ray virtual cluster for inference workers.
        generation_config: SGLang generation config (typed as SGLangConfig).

    Returns:
        Tuple of (SGLangGeneration instance, initialization wall time in seconds).
    """
    t0 = time.perf_counter()
    pg = SGLangGeneration(
        cluster=inference_cluster,
        sglang_cfg=generation_config,
    )
    pg.check_weights(action="snapshot")
    pg.check_weights(action="reset")
    pg.finish_generation()
    return pg, time.perf_counter() - t0


def initialize_generation_with_policy(
    init_generation_fn: Callable,
    init_policy_fn: Callable,
    init_time_key: str,
    colocated_inference: bool,
    worker_init_timing_metrics: dict,
    policy: Policy | None = None,
):
    """Initialize SGLang generation + policy, then run the weight-equality check.

    Verifier-only variant: after both sides are up and refit metadata is
    exchanged, this always refits the policy weights into SGLang and calls
    `check_weights(action="compare")` against the snapshot taken in
    `init_sglang`. Production GRPO uses its own copy in `grpo.setup`.

    Args:
        init_generation_fn: Callable returning (engine, init_time_s).
        init_policy_fn: Callable returning (policy, init_time_s).
        init_time_key: Metrics key for generation init time.
        colocated_inference: Whether inference is colocated with training.
        worker_init_timing_metrics: Dict populated with init/parallel timing.
        policy: Optional pre-initialized policy; if set, init_policy_fn is skipped.

    Returns:
        Tuple of (policy_generation, policy).
    """
    use_parallel_init = not colocated_inference and policy is None

    if use_parallel_init:
        print(
            "  ⚡ Using parallel worker initialization (non-colocated mode)",
            flush=True,
        )
        parallel_start_time = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as executor:
            generation_future = executor.submit(init_generation_fn)
            policy_future = executor.submit(init_policy_fn)
            policy_generation, generation_time = generation_future.result()
            policy, policy_time = policy_future.result()
        parallel_wall_time = time.perf_counter() - parallel_start_time

        worker_init_timing_metrics[init_time_key] = generation_time
        worker_init_timing_metrics["policy_init_time_s"] = policy_time
        worker_init_timing_metrics["parallel_wall_time_s"] = parallel_wall_time
        worker_init_timing_metrics["parallel_init_enabled"] = True
    else:
        print(
            "  ⚙️  Using sequential worker initialization (colocated mode)",
            flush=True,
        )
        policy_generation, generation_time = init_generation_fn()
        worker_init_timing_metrics[init_time_key] = generation_time

        if policy is None:
            policy, policy_time = init_policy_fn()
            worker_init_timing_metrics["policy_init_time_s"] = policy_time
        worker_init_timing_metrics["parallel_init_enabled"] = 0.0

    state_dict_info = policy.prepare_refit_info()
    policy_generation.prepare_refit_info(state_dict_info)

    refit_policy_generation(
        policy=policy,
        policy_generation=policy_generation,
        colocated_inference=colocated_inference,
    )
    policy_generation.check_weights(action="compare")
    policy_generation.finish_generation()
    policy.prepare_for_training()

    return policy_generation, policy


def parse_sglang_args():
    """Parse args for the sglang weight-check flow: --config + hydra-style overrides."""
    parser = argparse.ArgumentParser(
        description="SGLang weight-update verification via the same YAML config GRPO uses"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config file (same file used by run_grpo.py).",
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def main_sglang():
    """Load the GRPO YAML config and run the SGLang weight-equality check.

    Mirrors run_grpo.py's config-loading pipeline and grpo.setup's colocated
    cluster bootstrap, but skips dataset/loss/logger/checkpointer/grpo_state
    since the verifier only needs to refit weights once and diff against the
    pre-reset snapshot. Generation backend must be sglang.
    """
    args, overrides = parse_sglang_args()

    register_omegaconf_resolvers()
    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")
    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)
    master_config = OmegaConf.to_container(config, resolve=True)

    init_ray()

    tokenizer = get_tokenizer(master_config["policy"]["tokenizer"])

    policy_config = master_config["policy"]
    generation_config = master_config["policy"]["generation"]
    cluster_config = master_config["cluster"]
    assert generation_config is not None, "policy.generation must be set in the config"
    assert generation_config["backend"] == "sglang", (
        f"refit_verifier sglang mode requires backend=sglang, got "
        f"{generation_config['backend']!r}"
    )
    assert generation_config["colocated"]["enabled"], (
        "refit_verifier sglang mode currently only supports colocated inference"
    )

    generation_config = configure_generation_config(generation_config, tokenizer)
    generation_config["model_name"] = policy_config["model_name"]
    generation_config["sglang_cfg"].setdefault(
        "model_path", policy_config["model_name"]
    )

    policy_nodes = cluster_config["num_nodes"]
    policy_gpus_per_node = cluster_config["gpus_per_node"]
    cluster = RayVirtualCluster(
        name="refit_verifier_cluster",
        bundle_ct_per_node_list=[policy_gpus_per_node] * policy_nodes,
        use_gpus=True,
        num_gpus_per_node=policy_gpus_per_node,
        max_colocated_worker_groups=2,
    )
    print(
        f"  ✓ Ray cluster initialized with {policy_nodes} nodes "
        f"× {policy_gpus_per_node} GPUs/node",
        flush=True,
    )

    def init_policy_fn():
        t0 = time.perf_counter()
        p = Policy(
            cluster=cluster,
            config=policy_config,
            tokenizer=tokenizer,
            init_reference_model=False,
            init_optimizer=False,
        )
        return p, time.perf_counter() - t0

    worker_init_timing_metrics: dict = {}
    policy_generation, _ = initialize_generation_with_policy(
        init_generation_fn=lambda: init_sglang(cluster, generation_config),
        init_policy_fn=init_policy_fn,
        init_time_key="sglang_init_time_s",
        colocated_inference=True,
        worker_init_timing_metrics=worker_init_timing_metrics,
    )

    print("\n--- SGLang weight-check timing ---")
    for k, v in worker_init_timing_metrics.items():
        print(f"  {k}: {v}")

    policy_generation.shutdown()
    print("SGLang weight-check completed successfully!")


def _is_sglang_mode() -> bool:
    """Detect the sglang flow by the presence of --config in argv.

    The vLLM path has no --config argument, so its presence unambiguously
    signals the YAML-driven sglang flow. Inspecting sys.argv here keeps
    the existing vLLM path byte-identical.
    """
    import sys

    for tok in sys.argv[1:]:
        if tok == "--config" or tok.startswith("--config="):
            return True
    return False


def main_vllm():
    """VLLM weight-check flow."""
    args = parse_args()
    resolve_cluster_sizes(args)

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    if args.refit_transport == "vllm_http_sparse":
        require(
            args.non_colocated,
            "--refit_transport=vllm_http_sparse requires --non_colocated.",
        )
        require(
            args.enable_delta_compression,
            "--refit_transport=vllm_http_sparse requires --enable_delta_compression.",
        )
        require(
            args.delta_full_sync_interval > 1,
            "--refit_transport=vllm_http_sparse requires --delta_full_sync_interval > 1.",
        )
    if args.serve_vllm_http_refit:
        require(
            args.refit_transport == "vllm_http_sparse",
            "--serve_vllm_http_refit requires --refit_transport=vllm_http_sparse.",
        )
        require(
            not args.external_vllm_http_refit,
            "Use only one of --serve_vllm_http_refit or --external_vllm_http_refit.",
        )
    elif args.serve_vllm_http_refit_relay:
        raise ValueError(
            "--serve_vllm_http_refit_relay requires --serve_vllm_http_refit."
        )
    if args.external_vllm_http_refit:
        require(
            args.refit_transport == "vllm_http_sparse",
            "--external_vllm_http_refit requires --refit_transport=vllm_http_sparse.",
        )
        require(
            bool(_comma_separated_values(args.vllm_http_refit_urls)),
            "--external_vllm_http_refit requires explicit --vllm_http_refit_urls.",
        )
        require(
            args.skip_logprob_compare,
            "--external_vllm_http_refit cannot run local vLLM logprob comparison; pass --skip_logprob_compare.",
        )

    ray.init()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    megatron_cluster = None
    generation_cluster = None
    policy = None
    vllm_inference_policy = None
    try:
        megatron_config, vllm_config = setup_configs(args, tokenizer)
        if args.serve_vllm_http_refit:
            generation_cluster, vllm_inference_policy = setup_vllm_http_refit_server(
                args, vllm_config
            )
            if args.serve_seconds > 0:
                print(
                    f"Serving vLLM HTTP refit endpoints for {args.serve_seconds:.1f}s",
                    flush=True,
                )
                time.sleep(args.serve_seconds)
            else:
                print("Serving vLLM HTTP refit endpoints until interrupted", flush=True)
                while True:
                    time.sleep(3600)
            return

        (
            megatron_cluster,
            generation_cluster,
            policy,
            vllm_inference_policy,
        ) = setup_clusters_and_policies(args, megatron_config, vllm_config, tokenizer)

        print("\n--- Initializing weight synchronizer ---")
        refit_urls = _comma_separated_values(args.vllm_http_refit_urls)
        weight_sync = create_weight_synchronizer(
            policy=policy,
            generation=vllm_inference_policy,
            generation_backend=VLLM_BACKEND,
            colocated=not args.non_colocated,
            train_cluster=megatron_cluster,
            inference_cluster=generation_cluster,
            refit_buffer_size_gb=args.refit_buffer_size_gb,
            refit_transport=(args.refit_transport if args.non_colocated else None),
            refit_urls=refit_urls,
            refit_api_key_env_var=args.vllm_http_refit_api_key_env_var,
            refit_request_timeout_s=args.vllm_http_refit_timeout_s,
        )
        weight_sync.init_communicator()

        generation_data = prepare_input_data(args.prompt, tokenizer)
        first_pass_is_delta = maybe_overlap_initial_http_baseline(
            policy,
            weight_sync,
            refit_urls,
            args,
            generation_data,
            tokenizer,
        )
        run_model_refitting(
            policy,
            vllm_inference_policy,
            weight_sync,
            args,
            source_baseline_only_full_sync=(
                args.non_colocated
                and args.refit_transport == "vllm_http_sparse"
                and not first_pass_is_delta
            ),
            first_pass_is_delta=first_pass_is_delta,
        )

        if not args.skip_logprob_compare:
            vllm_logprobs_data, megatron_generation_data = (
                generate_and_compare_logprobs(
                    policy, vllm_inference_policy, generation_data
                )
            )
            analyze_logprob_differences(
                vllm_logprobs_data,
                megatron_generation_data,
                generation_data,
                tokenizer,
                args.prompt,
            )
    finally:
        cleanup_resources(
            policy,
            vllm_inference_policy,
            megatron_cluster,
            generation_cluster,
        )
        ray.shutdown()


def main():
    """Dispatch to the sglang or vllm weight-check flow."""
    if _is_sglang_mode():
        main_sglang()
    else:
        main_vllm()
    print("Script completed successfully!")


if __name__ == "__main__":
    main()
