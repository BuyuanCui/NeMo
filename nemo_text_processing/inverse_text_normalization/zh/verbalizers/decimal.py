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

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False

from nemo_text_processing.inverse_text_normalization.zh.graph_utils import (
    NEMO_DIGIT,
    GraphFst,
    delete_space,
    NEMO_NON_BREAKING_SPACE,
    NEMO_NOT_QUOTE,
)

class DecimalFst(GraphFst):
    def __init__(self):
        super().__init__(name="decimal",kind="verbalize")
        exactly_three_digits = NEMO_DIGIT **3
        at_most_three_digits = pynini.closure(NEMO_DIGIT,1,3) 
        
        space_every_three_integer = (
            at_most_three_digits + (pynutil.insert(NEMO_NON_BREAKING_SPACE) + exactly_three_digits).closure()
        )
        space_every_three_decimal = (
            pynini.accep(".") 
            + (exactly_three_digits + pynutil.insert(NEMO_NON_BREAKING_SPACE)).closure()
            + at_most_three_digits
        )
        group_by_threes = space_every_three_integer | space_every_three_decimal
        self.group_by_threes = group_by_threes
        
        optional_sign = pynini.closure(pynini.cross("negative: \"true\"", "-") + delete_space, 0, 1)
        
        integer = (
            pynutil.delete("integer_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
    
        integer = integer @ group_by_threes
        optional_integer = pynini.closure(integer + delete_space, 0, 1)
        
        fractional = (
            pynutil.insert(".") 
            + pynutil.delete("fractional_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        fractional = fractional @ group_by_threes
        optional_fractional = pynini.closure(fractional + delete_space,0,1)
        
        quantity = (
            pynutil.delete("quantity:")
            + delete_space + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE,1)
            + pynutil.delete("\"")
        )
        
        optional_quantity = pynini.closure(pynutil.insert(" ") + quantity + delete_space, 0, 1)
        
        graph = (optional_integer + optional_fractional + optional_quantity).optimize()
        self.numebrs = graph
        graph = optional_sign + graph
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
        
