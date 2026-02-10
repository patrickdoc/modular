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
"""Weight adapters for Qwen3Next hybrid Mamba2-MoE models.

Converts HuggingFace safetensor weights to MAX format:

1. Strips the ``model.`` prefix.
2. Stacks per-expert ``gate_proj`` / ``up_proj`` into a fused
   ``gate_up_proj`` tensor and transposes ``down_proj`` for
   ``grouped_matmul_ragged``.
3. Renames ``mlp.gate`` → ``mlp.experts.gate.gate_score`` for the router.
4. Maps ``linear_attn.*`` weight names to the MAX module tree.
"""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.graph.type import Shape
from max.graph.weights import WeightData, Weights
from max.pipelines.lib import PipelineConfig
from transformers import AutoConfig

# Simple name substitutions applied to every non-expert weight.
_NAME_MAPPING = {
    "model.": "",
    "mlp.gate.weight": "mlp.experts.gate.gate_score.weight",
}


def _weight_data_to_numpy(wd: WeightData) -> np.ndarray:
    if wd.dtype == DType.bfloat16:
        buf = Buffer.from_dlpack(wd.data)
        return buf.view(dtype=DType.uint16, shape=buf.shape).to_numpy()
    return np.from_dlpack(wd.data)  # type: ignore


def _numpy_to_weight_data(
    arr: np.ndarray, name: str, original_dtype: DType
) -> WeightData:
    if original_dtype == DType.bfloat16:
        buf = Buffer.from_numpy(arr)
        buf_bf16 = buf.view(dtype=DType.bfloat16, shape=buf.shape)
        return WeightData(
            data=buf_bf16,
            name=name,
            dtype=original_dtype,
            shape=Shape(buf_bf16.shape),
        )
    return WeightData.from_numpy(arr.copy(), name)


# Regex for per-expert weights
_EXPERT_RE = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight"
)


def convert_qwen3_next_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig,
    pipeline_config: PipelineConfig,
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Convert Qwen3Next safetensor weights to MAX format."""
    new_state_dict: dict[str, WeightData] = {}

    # {layer_idx: {expert_idx: {proj: WeightData}}}
    expert_weights: dict[int, dict[int, dict[str, WeightData]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    # Pass 1 – partition expert vs non-expert weights
    for hf_name, value in state_dict.items():
        match = _EXPERT_RE.match(hf_name)
        if match:
            layer_idx = int(match.group(1))
            expert_idx = int(match.group(2))
            proj_type = match.group(3)
            expert_weights[layer_idx][expert_idx][proj_type] = value.data()
        else:
            max_name = hf_name
            for before, after in _NAME_MAPPING.items():
                max_name = max_name.replace(before, after)
            new_state_dict[max_name] = value.data()

    # Pass 2 – stack expert weights
    for layer_idx in sorted(expert_weights.keys()):
        experts = expert_weights[layer_idx]
        num_experts = len(experts)
        original_dtype = experts[0]["gate_proj"].dtype

        gate_projs, up_projs, down_projs = [], [], []
        for eidx in range(num_experts):
            gate_projs.append(_weight_data_to_numpy(experts[eidx]["gate_proj"]))
            up_projs.append(_weight_data_to_numpy(experts[eidx]["up_proj"]))
            down_projs.append(_weight_data_to_numpy(experts[eidx]["down_proj"]))

        # gate/up: [num_experts, moe_dim, hidden] → stack → transpose →
        # concat → [num_experts, hidden, 2*moe_dim]
        stacked_gate = np.ascontiguousarray(
            np.transpose(np.stack(gate_projs, axis=0), (0, 2, 1))
        )
        stacked_up = np.ascontiguousarray(
            np.transpose(np.stack(up_projs, axis=0), (0, 2, 1))
        )
        gate_up = np.ascontiguousarray(
            np.concatenate([stacked_gate, stacked_up], axis=2)
        )
        gu_name = f"layers.{layer_idx}.mlp.experts.gate_up_proj"
        new_state_dict[gu_name] = _numpy_to_weight_data(
            gate_up, gu_name, original_dtype
        )

        # down: [num_experts, hidden, moe_dim] → transpose → [num_experts, moe_dim, hidden]
        stacked_down = np.ascontiguousarray(
            np.transpose(np.stack(down_projs, axis=0), (0, 2, 1))
        )
        dn_name = f"layers.{layer_idx}.mlp.experts.down_proj"
        new_state_dict[dn_name] = _numpy_to_weight_data(
            stacked_down, dn_name, original_dtype
        )

    return new_state_dict
