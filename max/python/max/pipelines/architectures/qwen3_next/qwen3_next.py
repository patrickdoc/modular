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
"""Qwen3Next hybrid Mamba2-MoE model for single and multi-GPU inference.

The model interleaves:
* **Full attention** layers (every ``full_attention_interval``-th layer) with
  partial RoPE, QK-norm, and KV-cache.
* **Linear attention** layers (Mamba2 SSM) on all other layers with causal
  conv1d, state-space computation, and gated output.

All layers share the same MoE feed-forward structure with 512 routed experts,
a shared expert, and a shared-expert scalar gate.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorType,
    TensorValue,
    TensorValueLike,
    Weight,
    ops,
)
from max.graph.quantization import QuantizationEncoding
from max.nn.legacy.comm import Signals
from max.nn.legacy.comm.allreduce import Allreduce
from max.nn.legacy.embedding import VocabParallelEmbedding
from max.nn.legacy.kv_cache import KVCacheParams, PagedCacheValues
from max.nn.legacy.layer import LayerList, Module
from max.nn.legacy.linear import MLP, ColumnParallelLinear, Linear
from max.nn.legacy.norm import RMSNorm
from max.nn.legacy.rotary_embedding import Llama3RotaryEmbedding
from max.nn.legacy.transformer import ReturnLogits
from max.nn.legacy.transformer.distributed_transformer import (
    forward_sharded_layers,
)
from max.pipelines.architectures.qwen3vl_moe.nn.moe import Qwen3VLMoE

from .layers.attention import Qwen3NextAttention
from .layers.linear_attention import Qwen3NextLinearAttention
from .model_config import Qwen3NextConfig


def distribute_value(
    v: TensorValue, devices: list[DeviceRef]
) -> list[TensorValue]:
    return [v.to(device) for device in devices]


# -------------------------------------------------------------------------
# Shared-expert gated MoE wrapper
# -------------------------------------------------------------------------


class Qwen3NextMoE(Module):
    """MoE block with routed experts, a shared expert, and a shared-expert gate.

    Weight mapping from HuggingFace::

        mlp.gate.weight                        → router
        mlp.experts.{j}.{gate,up,down}_proj    → routed experts (stacked by adapter)
        mlp.shared_expert.{gate,up,down}_proj  → shared expert MLP
        mlp.shared_expert_gate.weight          → sigmoid gate for shared expert
    """

    def __init__(
        self,
        config: Qwen3NextConfig,
        dtype: DType,
        devices: list[DeviceRef],
    ) -> None:
        super().__init__()
        self.devices = devices

        # Routed experts (uses stacked weight tensors via weight adapter)
        self.experts = Qwen3VLMoE(
            devices=devices,
            hidden_dim=config.hidden_size,
            num_experts=config.num_experts,
            num_experts_per_token=config.num_experts_per_tok,
            moe_dim=config.moe_intermediate_size,
            dtype=dtype,
        )

        # Shared expert (standard MLP)
        shared_dim = (
            config.shared_expert_intermediate_size
            or config.moe_intermediate_size
        )
        self.shared_expert = MLP(
            dtype,
            quantization_encoding=None,
            hidden_dim=config.hidden_size,
            feed_forward_length=shared_dim,
            devices=devices,
        )

        # Scalar gate for the shared expert: sigmoid(x @ W^T) → [seq, 1]
        self.shared_expert_gate = Linear(
            in_dim=config.hidden_size,
            out_dim=1,
            dtype=dtype,
            device=devices[0],
            has_bias=False,
        )

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return getattr(self, "_sharding_strategy", None)

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        self._sharding_strategy = strategy
        self.experts.sharding_strategy = strategy
        self.shared_expert.sharding_strategy = ShardingStrategy.replicate(
            strategy.num_devices
        )
        self.shared_expert_gate.sharding_strategy = ShardingStrategy.replicate(
            strategy.num_devices
        )

    def shard(self, devices: list[DeviceRef]) -> list[Qwen3NextMoE]:
        # Replicate the whole MoE block for now (TP sharding is future work)
        return [self] * len(devices)

    def __call__(self, x: TensorValue) -> TensorValue:
        # Routed expert output
        routed_out = self.experts(x)

        # Shared expert output, gated by sigmoid
        shared_out = self.shared_expert(x)
        gate = ops.sigmoid(self.shared_expert_gate(x))  # [seq, 1]
        return routed_out + gate * shared_out


# -------------------------------------------------------------------------
# Transformer block (one per layer)
# -------------------------------------------------------------------------


class Qwen3NextTransformerBlock(Module):
    """Single transformer block supporting either full or linear attention."""

    def __init__(
        self,
        config: Qwen3NextConfig,
        layer_idx: int,
        rope: Llama3RotaryEmbedding | None,
        create_norm: Callable[..., RMSNorm],
        linear_cls: Callable[..., Linear],
    ) -> None:
        super().__init__()
        self.devices = config.devices
        self.layer_idx = layer_idx
        num_devices = len(config.devices)
        self.is_full_attention = Qwen3NextConfig.is_full_attention_layer(
            layer_idx, config.full_attention_interval
        )

        # ---- attention (full or linear) ---------------------------------
        if self.is_full_attention:
            assert rope is not None
            self.self_attn = Qwen3NextAttention(
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                hidden_size=config.hidden_size,
                kv_params=config.kv_params,
                layer_idx=layer_idx,
                dtype=config.dtype,
                rope=rope,
                linear_cls=linear_cls,
                devices=config.devices,
                scale=config.attention_multiplier,
                has_bias=config.attention_bias,
            )
            self.self_attn.sharding_strategy = (
                ShardingStrategy.tensor_parallel(num_devices)
            )
            self.self_attn_shards = self.self_attn.shard(config.devices)
        else:
            self.linear_attn = Qwen3NextLinearAttention(
                config=config,
                layer_idx=layer_idx,
                dtype=config.dtype,
                devices=config.devices,
                linear_cls=linear_cls,
            )
            self.linear_attn.sharding_strategy = (
                ShardingStrategy.tensor_parallel(num_devices)
            )
            self.linear_attn_shards = self.linear_attn.shard(config.devices)

        # ---- MoE feed-forward -------------------------------------------
        self.mlp = Qwen3NextMoE(
            config=config, dtype=config.dtype, devices=config.devices
        )
        self.mlp.sharding_strategy = ShardingStrategy.tensor_parallel(
            num_devices
        )
        self.mlp_shards = self.mlp.shard(config.devices)

        # ---- norms (replicated) -----------------------------------------
        self.input_layernorm = create_norm()
        self.input_layernorm.sharding_strategy = ShardingStrategy.replicate(
            num_devices
        )
        self.input_layernorm_shards = self.input_layernorm.shard(config.devices)

        self.post_attention_layernorm = create_norm()
        self.post_attention_layernorm.sharding_strategy = (
            ShardingStrategy.replicate(num_devices)
        )
        self.post_attention_layernorm_shards = (
            self.post_attention_layernorm.shard(config.devices)
        )

        self.allreduce = Allreduce(num_accelerators=num_devices)

    def __call__(
        self,
        layer_idx: TensorValue,
        cache_idx: TensorValue | None,
        xs: list[TensorValue],
        kv_collections: list[PagedCacheValues],
        freqs_cis: list[TensorValue],
        input_row_offsets: list[TensorValue],
        signal_buffers: list[BufferValue],
    ) -> list[TensorValue]:
        """Forward pass.

        Args:
            layer_idx: Absolute layer index (uint32 scalar).
            cache_idx: KV-cache index for full-attention layers (None for linear).
            xs: Per-device hidden states.
            kv_collections: Per-device KV caches (used only by full-attn layers).
            freqs_cis: Per-device RoPE frequencies.
            input_row_offsets: Per-device row offsets.
            signal_buffers: Per-device signal buffers for allreduce.
        """
        # Pre-attention norm
        norm_xs = forward_sharded_layers(self.input_layernorm_shards, xs)

        # Attention
        if self.is_full_attention:
            assert cache_idx is not None
            attn_outs = [
                shard(
                    cache_idx,
                    norm_xs[i],
                    kv_collections[i],
                    freqs_cis[i],
                    input_row_offsets[i],
                )
                for i, shard in enumerate(self.self_attn_shards)
            ]
            if len(self.devices) > 1:
                attn_outs = self.allreduce(attn_outs, signal_buffers)
        else:
            attn_outs = [
                shard(norm_xs[i])
                for i, shard in enumerate(self.linear_attn_shards)
            ]
            # Linear attention shards are replicated, no allreduce needed

        # Residual
        hs = [x + a for x, a in zip(xs, attn_outs, strict=True)]

        # Post-attention norm → MoE
        norm_outs = forward_sharded_layers(
            self.post_attention_layernorm_shards, hs
        )
        mlp_outs = forward_sharded_layers(self.mlp_shards, norm_outs)

        # Residual
        return [h + m for h, m in zip(hs, mlp_outs, strict=True)]


# -------------------------------------------------------------------------
# Full model
# -------------------------------------------------------------------------


class Qwen3Next(Module):
    """Qwen3Next hybrid Mamba2-MoE model."""

    def __init__(self, config: Qwen3NextConfig) -> None:
        super().__init__()
        self.config = config
        self.devices = config.devices
        self.num_devices = len(config.devices)

        if config.model_quantization_encoding == QuantizationEncoding.GPTQ:
            raise NotImplementedError("GPTQ not supported for Qwen3Next")
        if config.model_quantization_encoding is not None:
            raise NotImplementedError("GGUFQ not supported for Qwen3Next")

        # RoPE for full-attention layers (partial rotation)
        rotary_dim = int(config.kv_params.head_dim * config.partial_rotary_factor)
        self.rope = Llama3RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            head_dim=rotary_dim,
            interleaved=config.interleaved_rope_weights,
            scaling_params=config.rope_scaling_params,
        )

        # Norm factory
        if config.norm_method != "rms_norm" or config.rms_norm_eps is None:
            raise ValueError("Qwen3Next requires RMSNorm.")
        create_norm = functools.partial(
            RMSNorm,
            config.hidden_size,
            dtype=config.norm_dtype or DType.float32,
            eps=config.rms_norm_eps,
            multiply_before_cast=False,
        )
        linear_cls = functools.partial(
            Linear, float8_config=config.float8_config
        )

        # Transformer layers
        self.layers = LayerList(
            [
                Qwen3NextTransformerBlock(
                    config=config,
                    layer_idx=i,
                    rope=self.rope if Qwen3NextConfig.is_full_attention_layer(
                        i, config.full_attention_interval
                    ) else None,
                    create_norm=create_norm,
                    linear_cls=linear_cls,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        # Final norm
        self.norm = create_norm()
        self.norm.sharding_strategy = ShardingStrategy.replicate(
            self.num_devices
        )
        self.norm_shards = self.norm.shard(config.devices)

        # Embedding and LM head (parallel, no-op on single GPU)
        embedding_dtype = config.dtype
        if config.float8_config and config.float8_config.embedding_output_dtype:
            embedding_dtype = config.float8_config.embedding_output_dtype

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            embedding_dtype,
            config.devices,
        )
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            embedding_dtype,
            devices=config.devices,
            tied_weight=(
                self.embed_tokens.weight if config.tie_word_embeddings else None
            ),
        )

        self.kv_params = config.kv_params
        self.return_logits = config.return_logits
        self.embedding_multiplier = config.embedding_multiplier
        self.full_attention_interval = config.full_attention_interval

    def __call__(
        self,
        tokens: TensorValueLike,
        kv_collections: list[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
        signal_buffers: list[BufferValue],
    ) -> tuple[TensorValue, ...]:
        # Embeddings
        h = self.embed_tokens(tokens, signal_buffers)
        if self.embedding_multiplier != 1.0:
            h = [hi * self.embedding_multiplier for hi in h]

        # Distribute auxiliary tensors
        freqs_cis = distribute_value(self.rope.freqs_cis, self.devices)
        input_row_offsets_list = distribute_value(
            input_row_offsets, self.devices
        )

        # Forward through layers
        full_attn_cache_idx = 0
        for idx, layer in enumerate(self.layers):
            layer_idx_t = ops.constant(idx, DType.uint32, device=DeviceRef.CPU())

            if Qwen3NextConfig.is_full_attention_layer(
                idx, self.full_attention_interval
            ):
                cache_idx_t = ops.constant(
                    full_attn_cache_idx, DType.uint32, device=DeviceRef.CPU()
                )
                h = layer(
                    layer_idx_t,
                    cache_idx_t,
                    h,
                    kv_collections,
                    freqs_cis,
                    input_row_offsets_list,
                    signal_buffers,
                )
                full_attn_cache_idx += 1
            else:
                h = layer(
                    layer_idx_t,
                    None,  # no cache for linear attention
                    h,
                    kv_collections,
                    freqs_cis,
                    input_row_offsets_list,
                    signal_buffers,
                )

        # Logits from last token
        h0 = h[0]
        last_token_indices = input_row_offsets[1:] - 1
        last_token_h = ops.gather(h0, last_token_indices, axis=0)
        last_token_distributed = distribute_value(last_token_h, self.devices)

        norm_last_token = forward_sharded_layers(
            self.norm_shards, last_token_distributed
        )
        last_logits = ops.cast(
            self.lm_head(norm_last_token, signal_buffers)[0],
            DType.float32,
        )

        # Variable / all logits
        logits = None
        offsets = None

        if self.return_logits == ReturnLogits.VARIABLE:
            return_n_logits_range = ops.range(
                start=return_n_logits[0],
                stop=0,
                step=-1,
                out_dim="return_n_logits_range",
                dtype=DType.int64,
                device=self.devices[0],
            )
            computed_offsets = (
                ops.unsqueeze(input_row_offsets[1:], -1) - return_n_logits_range
            )
            last_indices = ops.reshape(computed_offsets, shape=(-1,))
            variable_tokens = [
                ops.gather(hd, last_indices, axis=0) for hd in h
            ]
            variable_normed = forward_sharded_layers(
                self.norm_shards, variable_tokens
            )
            logits = ops.cast(
                self.lm_head(variable_normed, signal_buffers)[0],
                DType.float32,
            )
            offsets = ops.range(
                0,
                TensorValue(last_indices.shape[0]) + return_n_logits[0],
                return_n_logits[0],
                out_dim="logit_offsets",
                dtype=DType.int64,
                device=self.devices[0],
            )
        elif self.return_logits == ReturnLogits.ALL:
            all_normalized = forward_sharded_layers(self.norm_shards, h)
            logits = ops.cast(
                self.lm_head(all_normalized, signal_buffers)[0],
                DType.float32,
            )
            offsets = input_row_offsets

        if logits is not None and offsets is not None:
            return (last_logits, logits, offsets)
        return (last_logits,)

    def input_types(
        self, kv_params: KVCacheParams
    ) -> tuple[TensorType | BufferType, ...]:
        device_ref = self.devices[0]
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )

        kv_inputs = kv_params.get_symbolic_inputs()
        base_inputs: list[TensorType | BufferType] = [
            tokens_type,
            input_row_offsets_type,
            return_n_logits_type,
        ]
        signals = Signals(devices=self.devices)
        signal_buffer_types = signals.input_types()
        flattened_kv_types = [
            kv_type for sublist in kv_inputs for kv_type in sublist
        ]
        return tuple(base_inputs + signal_buffer_types + flattened_kv_types)
