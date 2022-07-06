# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.zh.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space
from nemo_text_processing.inverse_text_normalization.zh.taggers.cardinal import CardinalFst


try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False

class CardinalFst(GraphFst):
        def __init__(self):
            super().__init__(name="cardinal", kind="verbalize")
            
            #remove the negative attribute and leaves the sign if occurs
            optional_sign = pynini.closure(pynutil.delete("negative: ") + delete_space + pynutil.delete("\"") + pynini.accep("-") + pynutil.delete("\"") + delete_space)
            
            #remove integer aspect
            graph = (pynutil.delete("integer:") + delete_space + pynutil.delete("\"") + pynini.closure(NEMO_DIGIT, 1) + pynutil.delete("\""))
            graph = optional_sign + graph
            delete_tokens = self.delete_tokens(graph)
            self.fst = delete_tokens.optimize()