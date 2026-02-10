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

"""Full attention layer for Qwen3Next with partial RoPE and QK-norm.

Used on every ``full_attention_interval``-th layer (e.g. layers 3, 7, 11, …).
The remaining layers use the Mamba2 SSM linear attention instead.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue, ops
from max.nn.legacy.attention import MHAMaskVariant
from max.nn.legacy.attention.attention_with_rope import _compute_shard_range
from max.nn.legacy.kernels import (
    flash_attention_ragged,
    fused_qk_ragged_rope,
    fused_qkv_ragged_matmul,
    rms_norm_key_cache,
)
from max.nn.legacy.kv_cache import KVCacheParams, PagedCacheValues
from max.nn.legacy.layer import Module, Shardable
from max.nn.legacy.linear import Linear
from max.nn.legacy.norm import RMSNorm
from max.nn.legacy.rotary_embedding import RotaryEmbedding


class Qwen3NextAttention(Module, Shardable):
    """Full softmax attention for Qwen3Next with per-head QK-norm and partial RoPE.

    Identical to :class:`Qwen3Attention` in structure but parameterised for the
    Qwen3Next head dimensions and partial rotary embedding factor.
    """

    def __init__(
        self,
        *,
        rope: RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        layer_idx: int,
        dtype: DType = DType.float32,
        devices: list[DeviceRef],
        linear_cls: Callable[..., Linear] = Linear,
        scale: float | None = None,
        has_bias: bool = False,
        qk_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.rope = rope
        self.n_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.layer_idx = layer_idx
        self.kv_params = kv_params
        self.has_bias = has_bias
        self.devices = devices
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.linear_cls = linear_cls
        self.scale = (
            scale
            if scale is not None
            else math.sqrt(1.0 / self.kv_params.head_dim)
        )
        self.qk_norm_eps = qk_norm_eps
        self._sharding_strategy: ShardingStrategy | None = None

        if not self.kv_params.cache_strategy.uses_opaque():
            raise ValueError(
                f"{self.kv_params.cache_strategy} cache strategy not supported"
            )

        # Per-head QK-norm (same as Qwen3)
        self.q_norm = RMSNorm(
            self.kv_params.head_dim,
            dtype=dtype,
            eps=self.qk_norm_eps,
            multiply_before_cast=False,
        )
        self.k_norm = RMSNorm(
            self.kv_params.head_dim,
            dtype=dtype,
            eps=self.qk_norm_eps,
            multiply_before_cast=False,
        )

        q_weight_dim = self.kv_params.head_dim * num_attention_heads
        kv_weight_dim = self.kv_params.head_dim * num_key_value_heads

        self.q_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=q_weight_dim,
            dtype=dtype,
            device=devices[0],
            has_bias=has_bias,
        )
        self.k_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=kv_weight_dim,
            dtype=dtype,
            device=devices[0],
            has_bias=has_bias,
        )
        self.v_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=kv_weight_dim,
            dtype=dtype,
            device=devices[0],
            has_bias=has_bias,
        )
        self.o_proj = linear_cls(
            in_dim=q_weight_dim,
            out_dim=hidden_size,
            dtype=dtype,
            device=devices[0],
            has_bias=has_bias,
        )

    @property
    def wqkv(self) -> TensorValue:
        wq: TensorValue = self.q_proj.weight
        wk: TensorValue = self.k_proj.weight
        wv: TensorValue = self.v_proj.weight
        return ops.concat((wq, wk, wv)).to(self.devices[0])

    @property
    def wqkv_bias(self) -> TensorValue | None:
        if not self.has_bias:
            return None
        assert self.q_proj.bias is not None
        assert self.k_proj.bias is not None
        assert self.v_proj.bias is not None
        return ops.concat(
            (self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)
        ).to(self.devices[0])

    # -- sharding ---------------------------------------------------------

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        num_devices = strategy.num_devices
        if strategy.is_replicate:
            for m in (
                self.q_proj,
                self.k_proj,
                self.v_proj,
                self.o_proj,
                self.q_norm,
                self.k_norm,
            ):
                m.sharding_strategy = strategy
        elif strategy.is_tensor_parallel:
            for proj in (self.q_proj, self.k_proj, self.v_proj):
                proj.sharding_strategy = ShardingStrategy.rowwise(num_devices)
            self.o_proj.sharding_strategy = (
                ShardingStrategy.head_aware_columnwise(
                    num_devices, self.n_heads, self.kv_params.head_dim
                )
            )
            self.q_norm.sharding_strategy = ShardingStrategy.replicate(
                num_devices
            )
            self.k_norm.sharding_strategy = ShardingStrategy.replicate(
                num_devices
            )
        else:
            raise ValueError("Only tensor_parallel and replicate supported.")
        self._sharding_strategy = strategy

    def shard(self, devices: Iterable[DeviceRef]) -> list[Qwen3NextAttention]:
        if not self._sharding_strategy:
            raise ValueError("Set sharding_strategy before calling shard().")

        devices_list = list(devices)
        num_devices = len(devices_list)

        q_shards = self.q_proj.shard(devices_list)
        k_shards = self.k_proj.shard(devices_list)
        v_shards = self.v_proj.shard(devices_list)
        o_shards = self.o_proj.shard(devices_list)
        qn_shards = self.q_norm.shard(devices_list)
        kn_shards = self.k_norm.shard(devices_list)

        shards: list[Qwen3NextAttention] = []
        for idx, device in enumerate(devices_list):
            hs, he = _compute_shard_range(self.n_heads, idx, num_devices)
            kvs, kve = _compute_shard_range(
                self.num_key_value_heads, idx, num_devices
            )
            s = Qwen3NextAttention(
                rope=self.rope,
                num_attention_heads=he - hs,
                num_key_value_heads=kve - kvs,
                hidden_size=self.hidden_size,
                kv_params=self.kv_params,
                layer_idx=self.layer_idx,
                dtype=self.dtype,
                devices=[device],
                linear_cls=self.linear_cls,
                scale=self.scale,
                has_bias=self.has_bias,
                qk_norm_eps=self.qk_norm_eps,
            )
            s.q_proj = q_shards[idx]
            s.k_proj = k_shards[idx]
            s.v_proj = v_shards[idx]
            s.o_proj = o_shards[idx]
            s.q_norm = qn_shards[idx]
            s.k_norm = kn_shards[idx]
            shards.append(s)
        return shards

    # -- forward ----------------------------------------------------------

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        total_seq_len = x.shape[0]

        xq = fused_qkv_ragged_matmul(
            self.kv_params,
            input=x,
            wqkv=self.wqkv,
            bias=self.wqkv_bias,
            input_row_offsets=input_row_offsets,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
        )

        # Per-head QK-norm before RoPE
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))
        xq = self.q_norm(xq)

        rms_norm_key_cache(
            self.kv_params,
            kv_collection=kv_collection,
            gamma=self.k_norm.weight.cast(self.kv_params.dtype).to(
                self.devices[0]
            ),
            epsilon=self.qk_norm_eps,
            layer_idx=layer_idx,
            total_seq_len=total_seq_len,
            input_row_offsets=input_row_offsets,
            weight_offset=0.0,
            multiply_before_cast=False,
            per_head_norm=True,
        )

        # Partial RoPE: freqs_cis is already sized to the rotary subset
        freqs_cis = ops.cast(freqs_cis, xq.dtype).to(xq.device)
        xq = fused_qk_ragged_rope(
            self.kv_params,
            xq,
            input_row_offsets,
            kv_collection,
            freqs_cis,
            layer_idx,
            interleaved=self.rope.interleaved,
        )

        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=input_row_offsets,
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.scale,
        )

        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])
        return self.o_proj(attn_out)
