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


#%load_ext autoreload
#%autoreload 2

import pynini
import nemo_text_processing
from pynini.lib import pynutil

def apply_fst(text,fst):#applies this function for test purposes: apply_fst("the text",the fst_you_built)
    try:
        print(pynini.shortestpath(text @fst).string())
    except:
        print(f"Error: No valid output with given input: ' {text}'")

from nemo_text_processing.text_normalization.en.graph_utils import GraphFst, NEMO_DIGIT, delete_space, NEMO_SIGMA, NEMO_NOT_QUOTE, delete_extra_space, NEMO_NON_BREAKING_SPACE
from nemo_text_processing.text_normalization.normalize import Normalizer

from nemo_text_processing.inverse_text_normalization.zh.taggers.cardinal import CardinalFst
from nemo_text_processing.inverse_text_normalization.zh.taggers.decimal import DecimalFst
from nemo_text_processing.inverse_text_normalization.zh.taggers.money import MoneyFst
from nemo_text_processing.inverse_text_normalization.zh.taggers.ordinal import OrdinalFst
from nemo_text_processing.inverse_text_normalization.zh.taggers.punctuation import PunctuationFst
from nemo_text_processing.inverse_text_normalization.zh.taggers.time import TimeFst
from nemo_text_processing.inverse_text_normalization.zh.taggers.whitelist import WhiteListFst
from nemo_text_processing.inverse_text_normalization.zh.taggers.word import WordFst
from nemo_text_processing.inverse_text_normalization.zh.verbalizers.cardinal import CardinalFst
from nemo_text_processing.inverse_text_normalization.zh.verbalizers.decimal import DecimalFst
from nemo_text_processing.inverse_text_normalization.zh.verbalizers.money import MoneyFst
from nemo_text_processing.inverse_text_normalization.zh.verbalizers.ordinal import OrdinalFst
from nemo_text_processing.inverse_text_normalization.zh.verbalizers.time import TimeFst
from nemo_text_processing.inverse_text_normalization.zh.verbalizers.whitelist import WhiteListFst
from nemo_text_processing.inverse_text_normalization.zh.verbalizers.word import WordFst
from nemo_text_processing.inverse_text_normalization.zh.taggers.cardinal import CardinalFst

#One posisble way is to delete the morpheme第 and the add it back in the verbalizer
class OrdinalFst(GraphFst):
    def __init__(self, cardinal: GraphFst):
        super().__init__(name="ordinal", kind="classify")
        cardinals = cardinal.fst
        mandarin_morpheme = pynutil.delete("第") + cardinals
        graph_ordinal = pynutil.insert("integer: \"") + mandarin_morpheme + pynutil.insert("\"")
        graph_ordinal = self.add_tokens(graph_ordinal)
        self.fst = graph_ordinal.optimize()

#Second method, but would leave 第 outside of the bracket, as <ordinal { integer: "第cardinal { integer: "50" }" }>
class OrdinalFst(GraphFst):
    def __init__(self, cardinal: GraphFst):
        super().__init__(name="ordinal", kind="classify")
        graph_cardinals = cardinal.fst
        mandarin_morpheme = pynini.union("第") + graph_cardinals
        graph_ordinal_final = pynutil.insert("integer: \"") + mandarin_morpheme + pynutil.insert("\"")
        graph_ordinal_final = self.add_tokens(graph_ordinal_final)
        self.fst = graph_ordinal_final.optimize()
        
        

#A third way, seems to work fine but might have potential issues?
class OrdinalFst(GraphFst):
    def __init__(self, cardinal: GraphFst):
        super().__init__(name="ordinal", kind="classify")
        graph_cardinals = CardinalFst().just_cardinals
        mandarin_morpheme = pynini.union("第") + graph_cardinals
        graph_ordinal = mandarin_morpheme + NEMO_SIGMA
        graph_ordinal_final = pynutil.insert("integer: \"") + graph_ordinal + pynutil.insert("\"")
        graph_ordinal_final = self.add_tokens(graph_ordinal_final)
        self.fst = graph_ordinal_final.optimize()
        

#Method four
class OrdinalFst(GraphFst):
    def __init__(self, cardinal: GraphFst):
        super().__init__(name="ordinal", kind="classify")
        graph_cardinals = CardinalFst().just_cardinals
        strip_morpheme = pynutil.delete("第") 
        graph_strip_morpheme = strip_morpheme + NEMO_SIGMA
        strip_ordinals = graph_strip_morpheme @ graph_cardinals
        graph_ordinals= pynutil.insert("第") + strip_ordinals
        graph_ordinal_final = pynutil.insert("integer: \"") + graph_ordinals + pynutil.insert("\"")
        graph_ordinal_final = self.add_tokens(graph_ordinal_final)
        self.fst = graph_ordinal_final.optimize()

cardinal = CardinalFst()
ordinal = OrdinalFst(cardinal).fst
exmaple1 = "第五十"# only correct; expected output= 第50
exmaple2 = "五十"
example_1 = "五十第"
apply_fst(exmaple1, ordinal)
apply_fst(exmaple2, ordinal)
apply_fst(example_1,ordinal)

