# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Config for Qwen3Next hybrid Mamba2-MoE models."""

from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.legacy.kv_cache import KVCacheParams
from max.pipelines.lib import KVCacheConfig, PipelineConfig
from transformers.models.auto.configuration_auto import AutoConfig
from typing_extensions import Self, override

from ..qwen3.model_config import Qwen3Config


@dataclass(kw_only=True)
class Qwen3NextConfig(Qwen3Config):
    """Configuration for Qwen3Next hybrid attention models.

    Extends Qwen3Config with Mamba2 SSM linear attention parameters and
    hybrid attention scheduling (alternating full attention and linear
    attention layers).
    """

    # Hybrid attention scheduling
    full_attention_interval: int = 4
    """Every N-th layer uses full softmax attention; others use Mamba2 SSM."""

    # Mamba2 SSM / linear attention parameters
    linear_conv_kernel_dim: int = 4
    """Kernel size for the causal 1D convolution in linear attention layers."""

    linear_key_head_dim: int = 128
    """SSM state dimension (d_state) per group in linear attention."""

    linear_num_key_heads: int = 16
    """Number of B/C groups (n_groups) in the Mamba2 SSM."""

    linear_num_value_heads: int = 32
    """Number of SSM heads (n_heads) in the Mamba2 SSM."""

    linear_value_head_dim: int = 128
    """Per-head dimension (headdim) in the Mamba2 SSM."""

    partial_rotary_factor: float = 0.25
    """Fraction of head dimensions that receive RoPE in full attention layers."""

    shared_expert_intermediate_size: int = 0
    """Intermediate size for the shared expert MLP. 0 means same as moe_intermediate_size."""

    @staticmethod
    def is_full_attention_layer(
        layer_idx: int, full_attention_interval: int
    ) -> bool:
        """Return True if layer_idx should use full softmax attention."""
        return (layer_idx + 1) % full_attention_interval == 0

    @staticmethod
    def get_num_full_attention_layers(huggingface_config: AutoConfig) -> int:
        """Return the number of layers that use full softmax attention."""
        n_layers = huggingface_config.num_hidden_layers
        interval = getattr(huggingface_config, "full_attention_interval", 4)
        return sum(
            1
            for i in range(n_layers)
            if Qwen3NextConfig.is_full_attention_layer(i, interval)
        )

    @staticmethod
    def construct_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        """Build KV cache params sized only for the full-attention layers.

        Linear attention (Mamba2 SSM) layers maintain their own recurrent
        state and do not use the KV cache.
        """
        data_parallel_degree = pipeline_config.model.data_parallel_degree
        if data_parallel_degree > 1:
            raise ValueError(
                "Data parallelism is not supported for Qwen3Next models"
            )
        num_kv_layers = Qwen3NextConfig.get_num_full_attention_layers(
            huggingface_config
        )
        return KVCacheParams(
            dtype=cache_dtype,
            n_kv_heads=huggingface_config.num_key_value_heads,
            head_dim=huggingface_config.head_dim,
            num_layers=num_kv_layers,
            page_size=kv_cache_config.kv_cache_page_size,
            cache_strategy=kv_cache_config.cache_strategy,
            enable_prefix_caching=kv_cache_config.enable_prefix_caching,
            enable_kvcache_swapping_to_host=kv_cache_config.enable_kvcache_swapping_to_host,
            host_kvcache_swap_space_gb=kv_cache_config.host_kvcache_swap_space_gb,
            devices=devices,
            data_parallel_degree=data_parallel_degree,
        )

    @override
    @classmethod
    def initialize(cls, pipeline_config: PipelineConfig) -> Self:
        huggingface_config = pipeline_config.model.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required for '{pipeline_config.model.model_path}', "
                "but config could not be loaded."
            )
        return cls.initialize_from_config(pipeline_config, huggingface_config)

    @override
    @classmethod
    def initialize_from_config(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> Self:
        """Initialize a Qwen3NextConfig with all hybrid-attention parameters."""
        # Get base Qwen3 config (handles MoE params, KV params, etc.)
        base = Qwen3Config.initialize_from_config(
            pipeline_config, huggingface_config
        )

        # Re-compute KV params for hybrid model (only full-attn layers)
        kv_cache_config = pipeline_config.model.kv_cache
        cache_dtype = kv_cache_config.cache_dtype
        n_devices = len(pipeline_config.model.device_specs)
        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in pipeline_config.model.device_specs[:n_devices]
        ]
        hybrid_kv_params = Qwen3NextConfig.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=device_refs,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

        # Read Qwen3Next-specific config fields
        full_attention_interval = getattr(
            huggingface_config, "full_attention_interval", 4
        )
        linear_conv_kernel_dim = getattr(
            huggingface_config, "linear_conv_kernel_dim", 4
        )
        linear_key_head_dim = getattr(
            huggingface_config, "linear_key_head_dim", 128
        )
        linear_num_key_heads = getattr(
            huggingface_config, "linear_num_key_heads", 16
        )
        linear_num_value_heads = getattr(
            huggingface_config, "linear_num_value_heads", 32
        )
        linear_value_head_dim = getattr(
            huggingface_config, "linear_value_head_dim", 128
        )
        partial_rotary_factor = getattr(
            huggingface_config, "partial_rotary_factor", 0.25
        )
        shared_expert_intermediate_size = getattr(
            huggingface_config,
            "shared_expert_intermediate_size",
            base.moe_intermediate_size,
        )

        return cls(
            # Inherited from Llama3Config / Qwen3Config
            hidden_size=base.hidden_size,
            num_attention_heads=base.num_attention_heads,
            num_key_value_heads=base.num_key_value_heads,
            num_hidden_layers=base.num_hidden_layers,
            rope_theta=base.rope_theta,
            rope_scaling_params=base.rope_scaling_params,
            rms_norm_eps=base.rms_norm_eps,
            intermediate_size=base.intermediate_size,
            interleaved_rope_weights=base.interleaved_rope_weights,
            vocab_size=base.vocab_size,
            dtype=base.dtype,
            model_quantization_encoding=base.model_quantization_encoding,
            quantization_config=base.quantization_config,
            max_seq_len=base.max_seq_len,
            kv_params=hybrid_kv_params,
            attention_multiplier=base.attention_multiplier,
            embedding_multiplier=base.embedding_multiplier,
            residual_multiplier=base.residual_multiplier,
            devices=base.devices,
            clip_qkv=base.clip_qkv,
            use_subgraphs=base.use_subgraphs,
            dist_gemm_config=base.dist_gemm_config,
            # MoE parameters (from Qwen3Config)
            num_experts=base.num_experts,
            num_experts_per_tok=base.num_experts_per_tok,
            moe_intermediate_size=base.moe_intermediate_size,
            mlp_only_layers=base.mlp_only_layers,
            norm_topk_prob=base.norm_topk_prob,
            decoder_sparse_step=base.decoder_sparse_step,
            # Qwen3Next-specific
            full_attention_interval=full_attention_interval,
            linear_conv_kernel_dim=linear_conv_kernel_dim,
            linear_key_head_dim=linear_key_head_dim,
            linear_num_key_heads=linear_num_key_heads,
            linear_num_value_heads=linear_num_value_heads,
            linear_value_head_dim=linear_value_head_dim,
            partial_rotary_factor=partial_rotary_factor,
            shared_expert_intermediate_size=shared_expert_intermediate_size,
        )
