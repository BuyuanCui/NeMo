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

from nemo_text_processing.text_normalization.en.graph_utils import GraphFst #NEMO_DIGIT, delete_space, NEMO_SIGMA, NEMO_NOT_QUOTE, delete_extra_space, NEMO_NON_BREAKING_SPACE
#from nemo_text_processing.text_normalization.normalize import Normalizer

class OrdinalFst(GraphFst):
    def __init__(self, cardinal: GraphFst):
        super().__init__(name="ordinal", kind="classify")
        graph_cardinals = cardinal.fst
        mandarin_morpheme = pynini.union("第") + graph_cardinals
        graph_ordinal_final = pynutil.insert("integer: \"") + mandarin_morpheme + pynutil.insert("\"")
        graph_ordinal_final = self.add_tokens(graph_ordinal_final)
        self.fst = graph_ordinal_final.optimize()
            
