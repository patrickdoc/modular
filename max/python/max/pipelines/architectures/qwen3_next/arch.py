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

from max.graph.weights import WeightsFormat
from max.interfaces import PipelineTask
from max.nn.legacy.kv_cache import KVCacheStrategy
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    RopeType,
    SupportedArchitecture,
    SupportedEncoding,
    TextTokenizer,
)

from .model import Qwen3NextModel
from .model_config import Qwen3NextConfig
from .weight_adapters import convert_qwen3_next_state_dict

qwen3_next_arch = SupportedArchitecture(
    name="Qwen3NextForCausalLM_Legacy",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["Qwen/Qwen3-Coder-Next"],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.bfloat16: [
            KVCacheStrategy.PAGED,
        ],
        SupportedEncoding.float32: [
            KVCacheStrategy.PAGED,
        ],
    },
    pipeline_model=Qwen3NextModel,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    rope_type=RopeType.normal,
    weight_adapters={
        WeightsFormat.safetensors: convert_qwen3_next_state_dict,
    },
    config=Qwen3NextConfig,
    multi_gpu_supported=True,
)
