# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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

from nemo_text_processing.inverse_text_normalization.zh.utils import get_abs_path
from nemo_text_processing.inverse_text_normalization.zh.graph_utils import (
    NEMO_DIGIT,
    GraphFst,
)
from nemo_text_processing.inverse_text_normalization.zh.taggers.cardinal import CardinalFst


try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


def get_quantity(decimal, cardinal):
    suffix = pynini.union("万","十万","百万","千万","亿","十亿","百亿","千亿")
    numbers = cardinal
    res = (pynutil.insert("integer_part: \"") + numbers +pynutil.insert("\"") + pynutil.insert(" quantity: \"") + suffix + pynutil.insert("\"") )
    res = res | (res + decimal + pynutil.insert(" quantity: \"") + suffix + pynutil.insert("\""))
    return res

class DecimalFst(GraphFst):
    def __init__(self, cardinal: GraphFst):
        super().__init__(name="decimal",kind="classify")
        cardinal = cardinal.just_cardinals
        delete_decimal = pynutil.delete("点")
        
        graph_integer = pynutil.insert("integer_part: \"") + cardinal + pynutil.insert("\" ") 
        graph_integer_or_none = graph_integer | pynutil.insert("integer_part: \"0\" ", weight=0.01)
        
        graph_string_of_cardinals = cardinal
        graph_string_of_cardinals = pynini.closure(graph_string_of_cardinals,1)
        graph_fractional = pynutil.insert("fractional_part: \"") + graph_string_of_cardinals + pynutil.insert("\"")
        
        graph_decimal_no_sign = graph_integer_or_none + pynutil.delete("点") + graph_fractional
        
        graph_negative = pynini.cross("负", "negative: \"-\" ") 
        graph_negative = pynini.closure(graph_negative,0,1)
        
        graph_decimal = graph_negative + graph_decimal_no_sign
        graph_decimal = graph_decimal | (graph_negative + get_quantity(graph_decimal_no_sign, cardinal))
        
        final_graph = self.add_tokens(graph_decimal)
        self.fst = final_graph.optimize()
                                

