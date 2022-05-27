# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.fr.graph_utils import NEMO_NOT_QUOTE, GraphFst

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class TelephoneFst(GraphFst):
    """
    Finite state transducer for verbalizing telephone, e.g.
        telephone { number_part: "02 33 43 53 22" }
        -> 02 33 43 53 22 
    """

    def __init__(self):
        super().__init__(name="telephone", kind="verbalize")

        number_part = pynutil.delete("number_part: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        delete_tokens = self.delete_tokens(number_part)
        self.fst = delete_tokens.optimize()
