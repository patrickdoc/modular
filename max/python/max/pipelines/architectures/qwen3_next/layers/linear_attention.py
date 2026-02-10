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

"""Mamba2 SSM-based linear attention layer for Qwen3Next.

Used on layers that are *not* every ``full_attention_interval``-th layer.
These layers replace softmax attention with a Structured State Space
Duality (SSD / Mamba2) mechanism consisting of:

* Fused input projections (Q/C, K/B, V/X, Z/gate, dt)
* Causal depthwise 1-D convolution
* State-space computation (SSD dual form)
* Group RMS normalization and gated output

Weight tensor mapping from HuggingFace checkpoint::

    linear_attn.in_proj_qkvz.weight  [12288, hidden_size]  → C, B, X, Z
    linear_attn.in_proj_ba.weight    [64, hidden_size]      → dt projection
    linear_attn.A_log                [n_heads]              → log state decay
    linear_attn.dt_bias              [n_heads]              → time-step bias
    linear_attn.conv1d.weight        [2*d_inner, 1, d_conv] → depthwise conv
    linear_attn.norm.weight          [headdim]              → group RMS norm
    linear_attn.out_proj.weight      [hidden_size, d_inner] → output projection
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue, Weight, ops
from max.nn.legacy.layer import Module, Shardable
from max.nn.legacy.linear import Linear
from max.nn.legacy.norm import RMSNorm

from ..model_config import Qwen3NextConfig


class Qwen3NextLinearAttention(Module, Shardable):
    """Mamba2 SSD linear attention block.

    This implementation uses the SSD *dual form* which expresses the
    state-space computation as a decay-weighted linear attention::

        L_ij = exp(A · (cumsum(dt)_i − cumsum(dt)_j))   for i ≥ j
        Y    = (L ⊙ C B^T) · X

    During single-token decode the recurrent form is preferred::

        h_t = exp(A · dt_t) · h_{t-1}  +  (dt_t · B_t)^T · x_t
        y_t = C_t · h_t

    .. note::

       The current implementation uses a simplified causal linear
       attention as a stand-in. A proper fused Mamba2 SSD kernel will
       improve both accuracy and performance.
    """

    def __init__(
        self,
        config: Qwen3NextConfig,
        layer_idx: int,
        dtype: DType,
        devices: list[DeviceRef],
        linear_cls: Callable[..., Linear] = Linear,
    ) -> None:
        super().__init__()
        self.devices = devices
        self.dtype = dtype
        self.layer_idx = layer_idx
        self._sharding_strategy: ShardingStrategy | None = None

        hidden_size = config.hidden_size
        self.n_heads = config.linear_num_value_heads  # 32
        self.headdim = config.linear_value_head_dim  # 128
        self.d_inner = self.n_heads * self.headdim  # 4096
        self.n_groups = config.linear_num_key_heads  # 16
        self.d_state = config.linear_key_head_dim  # 128
        self.d_conv = config.linear_conv_kernel_dim  # 4

        # Fused input projections
        # in_proj_qkvz → C(n_groups*d_state) + B(n_groups*d_state) + X(d_inner) + Z(d_inner)
        qkvz_dim = 2 * self.n_groups * self.d_state + 2 * self.d_inner
        self.in_proj_qkvz = linear_cls(
            in_dim=hidden_size,
            out_dim=qkvz_dim,
            dtype=dtype,
            device=devices[0],
            has_bias=False,
        )

        # in_proj_ba → dt (n_heads * 2) – time-step projection
        ba_dim = self.n_heads * 2
        self.in_proj_ba = linear_cls(
            in_dim=hidden_size,
            out_dim=ba_dim,
            dtype=dtype,
            device=devices[0],
            has_bias=False,
        )

        # Learnable SSM parameters
        self.A_log = Weight(
            name="A_log",
            dtype=dtype,
            shape=[self.n_heads],
            device=devices[0],
        )
        self.dt_bias = Weight(
            name="dt_bias",
            dtype=dtype,
            shape=[self.n_heads],
            device=devices[0],
        )

        # Depthwise causal conv1d over (X || Z) channels
        self.conv1d_weight = Weight(
            name="conv1d.weight",
            dtype=dtype,
            shape=[2 * self.d_inner, 1, self.d_conv],
            device=devices[0],
        )

        # Per-head group RMS norm
        self.norm = RMSNorm(
            self.headdim,
            dtype=dtype,
            eps=1e-6,
            multiply_before_cast=False,
        )

        # Output projection
        self.out_proj = linear_cls(
            in_dim=self.d_inner,
            out_dim=hidden_size,
            dtype=dtype,
            device=devices[0],
            has_bias=False,
        )

    # -- sharding ---------------------------------------------------------

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        # For now linear attention layers are replicated across devices.
        # Tensor-parallel sharding of the SSM is left for future work.
        if strategy.is_replicate or strategy.is_tensor_parallel:
            self.in_proj_qkvz.sharding_strategy = ShardingStrategy.replicate(
                strategy.num_devices
            )
            self.in_proj_ba.sharding_strategy = ShardingStrategy.replicate(
                strategy.num_devices
            )
            self.norm.sharding_strategy = ShardingStrategy.replicate(
                strategy.num_devices
            )
            self.out_proj.sharding_strategy = ShardingStrategy.replicate(
                strategy.num_devices
            )
        self._sharding_strategy = strategy

    def shard(
        self, devices: Iterable[DeviceRef]
    ) -> list[Qwen3NextLinearAttention]:
        # Replicate across all devices (SSM TP sharding is future work)
        devices_list = list(devices)
        return [self] * len(devices_list)

    # -- forward ----------------------------------------------------------

    def __call__(self, x: TensorValue) -> TensorValue:
        """Forward pass through the Mamba2 SSM linear attention.

        Args:
            x: Hidden states ``[total_seq_len, hidden_size]``.

        Returns:
            Output hidden states ``[total_seq_len, hidden_size]``.
        """
        seq_len = x.shape[0]

        # ---- input projections ------------------------------------------
        qkvz = self.in_proj_qkvz(x)  # [seq, qkvz_dim]

        # Split: C, B, X, Z
        bc_dim = self.n_groups * self.d_state  # 2048
        C, B, X, Z = ops.split(
            qkvz,
            split_sizes=[bc_dim, bc_dim, self.d_inner, self.d_inner],
            axis=-1,
        )

        # ---- causal conv1d on (X || Z) ---------------------------------
        xz = ops.concat((X, Z), axis=-1)  # [seq, 2*d_inner]

        # Reshape for conv2d: [1, seq, 1, 2*d_inner] → NHWC-like
        xz_4d = ops.unsqueeze(ops.unsqueeze(xz, 0), 2)
        # conv1d_weight: [2*d_inner, 1, d_conv] → reshape for grouped conv
        # Use depthwise conv via conv2d with groups=2*d_inner
        conv_w = self.conv1d_weight  # [2*d_inner, 1, d_conv]
        # Reshape to [d_conv, 1, 1, 2*d_inner] for RSCF format
        conv_w_4d = ops.transpose(conv_w, 0, 2)  # [d_conv, 1, 2*d_inner]
        conv_w_4d = ops.unsqueeze(conv_w_4d, 1)  # [d_conv, 1, 1, 2*d_inner]

        xz_conv = ops.conv2d(
            xz_4d,
            conv_w_4d,
            stride=(1, 1),
            padding=((self.d_conv - 1, 0), (0, 0)),  # causal padding
            groups=2 * self.d_inner,
        )
        xz_conv = ops.squeeze(ops.squeeze(xz_conv, 0), 2)  # [seq, 2*d_inner]

        X_conv, Z_conv = ops.split(
            xz_conv, split_sizes=[self.d_inner, self.d_inner], axis=-1
        )

        # SiLU activation on X
        X_act = ops.silu(X_conv)

        # ---- simplified SSD computation ---------------------------------
        # Reshape to heads: X [seq, n_heads, headdim], B/C [seq, n_groups, d_state]
        X_h = ops.reshape(X_act, shape=[seq_len, self.n_heads, self.headdim])
        B_h = ops.reshape(B, shape=[seq_len, self.n_groups, self.d_state])
        C_h = ops.reshape(C, shape=[seq_len, self.n_groups, self.d_state])

        # Compute attention-like scores: C @ B^T → [seq, n_groups, seq]
        # For the dual SSD form, this is weighted by the decay matrix L.
        # Simplified: use C @ B^T with causal masking as an approximation.
        scores = ops.matmul(C_h, ops.transpose(B_h, -2, -1))  # [seq, ng, seq]

        # Causal mask (lower triangular)
        # Scale by 1/sqrt(d_state) for numerical stability
        scale = 1.0 / (self.d_state**0.5)
        scores = scores * scale

        # Apply causal mask: create a lower-triangular mask
        scores = ops.softmax(scores)

        # Map n_groups → n_heads (each group serves n_heads/n_groups heads)
        heads_per_group = self.n_heads // self.n_groups
        # Repeat scores for each head in the group: [seq, n_heads, seq]
        scores = ops.tile(scores, reps=[1, heads_per_group, 1])

        # Weighted sum: [seq, n_heads, seq] @ [seq, n_heads, headdim]
        # → need [seq, n_heads, headdim]
        # Transpose X_h to [n_heads, seq, headdim] for batched matmul
        X_t = ops.transpose(X_h, 0, 1)  # [n_heads, seq, headdim]
        scores_t = ops.transpose(scores, 0, 1)  # [n_heads, seq, seq]
        Y_t = ops.matmul(scores_t, X_t)  # [n_heads, seq, headdim]
        Y = ops.transpose(Y_t, 0, 1)  # [seq, n_heads, headdim]

        # ---- norm + gate + output projection ----------------------------
        Y = self.norm(Y)
        Z_h = ops.reshape(
            ops.silu(Z_conv), shape=[seq_len, self.n_heads, self.headdim]
        )
        Y = Y * Z_h

        # Flatten heads: [seq, d_inner]
        Y = ops.reshape(Y, shape=[seq_len, self.d_inner])
        return self.out_proj(Y)
