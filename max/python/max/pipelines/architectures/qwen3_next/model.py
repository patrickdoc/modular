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

from __future__ import annotations

import logging
from typing import Any, Literal

from max._core.engine import Model
from max.engine import InferenceSession
from max.graph import Graph
from max.graph.weights import Weights, WeightsAdapter
from max.pipelines.lib.interfaces import AlwaysSignalBuffersMixin

from ..llama3.model import LlamaModelBase
from .model_config import Qwen3NextConfig
from .qwen3_next import Qwen3Next

logger = logging.getLogger("max.pipelines")


class Qwen3NextModel(AlwaysSignalBuffersMixin, LlamaModelBase):
    """Pipeline model for Qwen3Next hybrid Mamba2-MoE architecture."""

    model: Model
    norm_method: Literal["rms_norm"] | Literal["layer_norm"] = "rms_norm"
    attention_bias: bool = False
    state_dict: dict[str, Any]

    def _build_graph(
        self,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        session: InferenceSession | None = None,
    ) -> Graph:
        state_dict = self._get_state_dict(weights, adapter)
        model_config = Qwen3NextConfig.initialize_from_config(
            self.pipeline_config, self.huggingface_config
        )
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            return_logits=self.return_logits,
            norm_method=self.norm_method,
            attention_bias=self.attention_bias,
        )

        nn_model = Qwen3Next(model_config)
        graph_inputs = nn_model.input_types(self.kv_params)

        nn_model.load_state_dict(
            state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=(
                not getattr(
                    self.huggingface_config, "tie_word_embeddings", False
                )
            ),
        )
        self.state_dict = nn_model.state_dict()

        num_devices = len(self.devices)

        with Graph("qwen3_next", input_types=graph_inputs) as graph:
            tokens, input_row_offsets, return_n_logits, *variadic_args = (
                graph.inputs
            )
            signal_buffers = [v.buffer for v in variadic_args[:num_devices]]
            kv_cache_inputs = variadic_args[num_devices:]
            kv_collections = self._unflatten_kv_inputs(kv_cache_inputs)

            outputs = nn_model(
                tokens.tensor,
                kv_collections,
                return_n_logits.tensor,
                input_row_offsets.tensor,
                signal_buffers,
            )
            graph.output(*outputs)
            return graph
