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


from nemo_text_processing.inverse_text_normalization.zh.utils import get_abs_path, num_to_word
from nemo_text_processing.inverse_text_normalization.zh.graph_utils import (
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    delete_space,
    Mandarin_Digit,
)

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals
        e.g. minus twenty three -> cardinal { integer: "23" negative: "-" } }
    Numbers below thirteen are not converted. 
    """

    def __init__(self):
        super().__init__(name="cardinal", kind="classify")
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))
        graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv"))
        #below：10-99的数列
        graph_tens_component = (graph_ties + graph_digit) | pynutil.insert("0") 
        graph_tens_component = pynini.union(graph_tens_component, graph_teen)
        #below: hundreds
        graph_hundred = pynini.cross("百", "")
        #graph_hundred_component = pynini.union(graph_digit + delete_space + graph_hundred, pynutil.insert("0"))
        graph_hundred_component = pynini.union(graph_digit + graph_hundred, pynutil.insert("00"))
        
        #graph_hundred_component += delete_space
        #graph_hundred_component += pynini.union(
        #    graph_teen | pynutil.insert("00"),
        #    (graph_ties | pynutil.insert("0")) + delete_space + (graph_digit | pynutil.insert("0")),
        #)
        graph_hundred_component += pynini.union(
            graph_teen | pynutil.insert("0"),
            (graph_ties | pynutil.insert("0")) + (graph_digit | pynutil.insert("00")),
        )
        #teen+insert zero or ties+insert zero+delete space+(gitis+insert zero) （下に融合）
        #graph_hundred_component_at_least_one_none_zero_digit = graph_hundred_component @ (
        #    pynini.closure(NEMO_DIGIT) + (NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT)
        #)
        graph_hundred_component_at_least_one_none_zero_digit = graph_hundred_component @ (
            pynini.closure(NEMO_DIGIT) + (NEMO_DIGIT, "0") + pynini.closure(NEMO_DIGIT)
        )

        #self.graph_hundred_component_at_least_one_none_zero_digit = (
        #    graph_hundred_component_at_least_one_none_zero_digit
        )
        graph_thousands = pynini.cross("千","")
        graph_thousands_component = pynini.union(graph_digit + graph_thousands, pynutil.insert("000"))
        graph_thousands_component += pynini.union(
            graph_teen | pyntil.insert("000"), (graph_ties | pynutil.insert("000")) + (graph_digit | pynutil.insert ("000")),
        )

        # tens as 10-99
        graph_tens_of_hundreds = pynini.cross("万","")
        graph_tens_of_hundreds_component = pynini.union(graph_digit + graph_tens_of_hundreds, pynutil.insert("0000"))
        graph_tens_of_hundreds_component += pynini.union(graph_teen | pynutil.insert ("000"), (graph_ties | pynutil.insert("000"), graph_digit | pynutil.insert("000")),
        )
        
        graph_tens_of_hundreds_component_at_least_one_none_zero_digit = graph_tens_of_hundreds_component @(
            pynini.closure(NEMO_DIGIT) + (NEMO_DIGIT, "0") + (NEMO_DIGIT, "0") + pynini.closure(NEMO_DIGIT)
        )
        self.graph_tens_of_hundreds_component_at_least_one_none_zero_digit = (
            graph_tens_of_hundreds_component_at_least_one_none_zero_digit
        )
        #Final for graph hundred

        #以下は大きいナンバーのグラフ（グラフ百に基づいて）

        #以下已移除delete_space     
        #graph_thousands = pynini.union(
        #    graph_hundred_component_at_least_one_none_zero_digit + pynutil.delete("千"),
        #    pynutil.insert("000", weight=0.1),
        #)
        graph_tenthousands = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + pynutil.delete("万"),
            pynutil.insert("0000", weight=0.1),
        )
        graph_hundredthousands = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + pynutil.delete("十万"),
            pynutil.insert("000", weight=0.1),
        )
        graph_million = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + pynutil.delete("百万"),
            pynutil.insert("000", weight=0.1),
        )
        graph_tenmillion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + pynutil.delete("千万"),
            pynutil.insert("000", weight=0.1),
        )
        graph_hundredmillion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + pynutil.delete("亿"),
            pynutil.insert("000", weight=0.1),
        )
        graph_billion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + pynutil.delete("十亿"),
            pynutil.insert("000", weight=0.1),
        )
        graph_tenbillion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + pynutil.delete("百亿"),
            pynutil.insert("000", weight=0.1),
        )
        graph_hundredbillion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + pynutil.delete("千亿"),
            pynutil.insert("000", weight=0.1),
        )
        graph_trillion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + pynutil.delete("兆"),
            pynutil.insert("000", weight=0.1),
        )
        graph_tentrillion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + pynutil.delete("十兆"),
            pynutil.insert("000", weight=0.1),
        )
        graph_hundredtrillion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + pynutil.delete("百兆"),
            pynutil.insert("000", weight=0.1),
        )
        graph_quadrillion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + pynutil.delete("千兆"),
            pynutil.insert("000", weight=0.1),
        )
        graph_giga = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + pynutil.delete("京"),
            pynutil.insert("000", weight=0.1),
        )
        graph_tengiga = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + pynutil.delete("十京"),
            pynutil.insert("000", weight=0.1),
        )
        graph_quintillion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + pynutil.delete("百京"),
            pynutil.insert("000", weight=0.1),
        )
        graph_thousandgiga = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + pynutil.delete("千京"),
            pynutil.insert("000", weight=0.1),
        )
        graph_gai = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + pynutil.delete("垓"),
            pynutil.insert("000", weight=0.1),
        )
        graph_sextillion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + pynutil.delete("十垓"),
            pynutil.insert("000", weight=0.1),
        )
        graph_hundredgai = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + pynutil.delete("百垓"),
            pynutil.insert("000", weight=0.1),
        )
        graph_thousandgai = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + pynutil.delete("千垓"),
            pynutil.insert("000", weight=0.1),
        )
        graph_septillion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + pynutil.delete("秭"),
            pynutil.insert("000", weight=0.1),
        )
        graph_tensextillion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + pynutil.delete("十秭"),
            pynutil.insert("000", weight=0.1),
        )
        graph_hundredsextillion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + pynutil.delete("百秭"),
            pynutil.insert("000", weight=0.1),
        )
        graph_thousandsextillion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + pynutil.delete("千垓"),
            pynutil.insert("000", weight=0.1),
        )
        #すべて融合
        graph = pynini.union(
            graph_thousandsextillion
            #+ delete_space
            + graph_hundredsextillion
            #+ delete_space
            + graph_tensextillion
            #+ delete_space
            + graph_septillion
            #+ delete_space
            + graph_thousandgai
            #+ delete_space
            + graph_hundredgai
            #+ delete_space
            + graph_sextillion
            #+ delete_space
            + graph_gai
            #+ delete_space
            + graph_thousandgiga
            #+ delete_space
            + graph_quintillion
            #+ delete_space
            + graph_tengiga
            #+ delete_space
            + graph_giga
            #+ delete_space
            + graph_quadrillion
            #+ delete_space
            + graph_hundredtrillion
            #+ delete_space
            + graph_tentrillion
            #+ delete_space
            + graph_trillion
            #+ delete_space
            + graph_hundredbillion
            #+ delete_space
            + graph_tenbillion
            #+ delete_space
            + graph_billion
            #+ delete_space
            + graph_hundredmillion
            #+ delete_space
            + graph_tenmillion
            #+ delete_space
            + graph_million
            #+ delete_space
            + graph_hundredthousands
            #+ delete_space
            + graph_tenthousands
            #+ delete_space
            + graph_thousands
            #+ delete_space
            + graph_hundred_component,
            graph_zero,
        )
        #redefine the graph just combined
        graph = graph @ pynini.union(
            pynutil.delete(pynini.closure("0")) + pynini.difference(NEMO_DIGIT, "0") + pynini.closure(NEMO_DIGIT), "0"
        )
        #delete all zeros+accept all except NEMO_DTGIT +delete all zeros, zero
        labels_exception = [num_to_word(x) for x in range(0, 13)]
        #for x in range (0,13) apply function num_to_word()
        graph_exception = pynini.union(*labels_exception)

        #graph = (
        #    pynini.cdrewrite(pynutil.delete("and"), NEMO_SPACE, NEMO_SPACE, NEMO_SIGMA)
        #    @ (NEMO_ALPHA + NEMO_SIGMA)
        #    @ graph
        #)
        
        self.graph_no_exception = graph

        self.graph = (pynini.project(graph, "input") - graph_exception.arcsort()) @ graph

        optional_minus_graph = pynini.closure(
            pynutil.insert("负: ") + pynini.cross("负", "\"-\"") + NEMO_SPACE, 0, 1
        )
        #cover finals number
        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
