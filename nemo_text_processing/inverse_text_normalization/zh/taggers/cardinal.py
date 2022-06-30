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
)

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


def apply_fst(text,fst):#applies this function for test purposes: apply_fst("the text",the fst_you_built)
    try:
        print(pynini.shortestpath(text @fst).string())
    except:
        print(f"Error: No valid output with given input: ' {text}'")

class CardinalFst(GraphFst):
    def __init__(self):
        super().__init__(name="cardinal", kind="classify") 
        
        #Cardinal gramamrs
        #Grammar for cardinals from 0-99
        zero = pynini.string_map(["零","0"])
        digits = pynini.string_map([("一","1"),("幺","1"),("壹","1"),
                         ("二","2"),("两","2"),("兩","2"),("貳","2"),
                         ("三","3"),("參","3"),
                         ("四","4"),("肆","4"),
                         ("五","5"),("伍","5"),
                         ("六","6"),("陸","6"),
                         ("七","7"),("柒","7"),
                         ("八","8"),("捌","8"),
                         ("九","9"),("玖","9")])
        
        graph_digits = digits | pynutil.insert("0")
        tens = pynini.string_map([("十","1"),("拾","1")])
        graph_tens = tens+graph_digits
        graph_all = graph_tens | zero   #leading zero issue solved later
        tens = pynini.string_map([("十","1"),("拾","1"),("一十","1"),("壹拾","1"),
                        ("二十","2"),("貳拾","2"),
                        ("三十","3"),("叁拾","3"),
                        ("四十","4"),("肆拾","4"),
                        ("五十","5"),("伍拾","5"),
                        ("六十","6"),("陸拾","6"),
                        ("七十","7"),("柒拾","7"),
                        ("八十","8"),("捌拾","8"),
                        ("九十","9"),("玖拾","9")])
        tens = tens | pynutil.insert("0") 
        graph_tens = tens + graph_digits
        graph_all = graph_tens | zero
        
        #Grammar for cardinals from 100-999
        delete_hundreds = pynutil.delete("百") | pynutil.delete("佰")
        delete_tens = pynutil.delete("十") | pynutil.delete("拾")
        delete_zero = pynutil.delete("零")
        graph_hundreds = ((graph_digits + delete_hundreds + graph_all) | (graph_digits + delete_hundreds + delete_zero + graph_all)) | pynutil.insert("000")
        
        #Grammars for cardinals from 1000-9999
        delete_thousands = pynutil.delete("千") | pynutil.delete("仟")
        graph_thousands = ((graph_digits + delete_thousands + graph_hundreds) | (graph_digits + delete_thousands + delete_zero + pynutil.insert("0") + graph_all)) | pynutil.insert("0000")
        
        #Grammars for cardinals from 10000-99999
        delete_ten_thousands = pynutil.delete("萬") | pynutil.delete("万")
        graph_ten_thousands = ((graph_digits + delete_ten_thousands + graph_thousands) | (graph_digits + delete_ten_thousands + delete_zero + pynutil.insert("0") + graph_hundreds) | (graph_digits + delete_ten_thousands + delete_zero + pynutil.insert("00") + graph_all))| pynutil.insert("00000")
        
        #Grammer for 100000-999999 (hundred thousands-十万)
        graph_hundred_thousands =  (tens + graph_ten_thousands) | pynutil.insert("000000")
        
        #grammar for millions 百万
        graph_millions = ((graph_hundreds + delete_ten_thousands + graph_thousands) |(graph_hundreds + delete_ten_thousands + delete_zero + pynutil.insert("000") + graph_digits) | (graph_hundreds + delete_ten_thousands + delete_zero + pynutil.insert("00") + graph_all) |(graph_hundreds + delete_ten_thousands + delete_zero + pynutil.insert("0") + graph_hundreds)) | pynutil.insert("0000000") 
        
        #grammar for ten millions 千万
        graph_ten_millions = ((graph_thousands + delete_ten_thousands + graph_thousands) | (graph_thousands + delete_ten_thousands + delete_zero + pynutil.insert("00") + graph_all) | (graph_thousands + delete_ten_thousands + delete_zero + pynutil.insert("0") + graph_hundreds)) | pynutil.insert("00000000")
        
        #grammar for hundred millions 亿
        delete_hundred_millions = pynutil.delete("亿") | pynutil.delete ("億")
        graph_hundred_millions = ((graph_digits + delete_hundred_millions + graph_ten_millions) | (graph_digits + delete_hundred_millions + delete_zero + pynutil.insert("0") + graph_millions) | (graph_digits + delete_hundred_millions + delete_zero + pynutil.insert("00") + graph_hundred_thousands)) | pynutil.insert("000000000")
        
        #grammars for billions 十億
        graph_billions = ((tens + delete_hundred_millions + graph_hundred_millions) | (graph_all + delete_hundred_millions + graph_ten_millions) | (graph_all + delete_hundred_millions +delete_zero + pynutil.insert("0") + graph_millions) | (graph_all + delete_hundred_millions +delete_zero + pynutil.insert("00") + graph_hundred_thousands) | (graph_all + delete_hundred_millions +delete_zero + pynutil.insert("000") + graph_ten_thousands) | (graph_all + delete_hundred_millions +delete_zero + pynutil.insert("0000") + graph_thousands) | (graph_all + delete_hundred_millions +delete_zero + pynutil.insert("00000") + graph_hundreds) | (graph_all + delete_hundred_millions +delete_zero + pynutil.insert("000000") + graph_all)) | pynutil.insert("0000000000")
        
        #grammars for ten billions 百億
        graph_ten_billions = (graph_hundreds + delete_hundred_millions + graph_ten_millions) | (graph_hundreds + delete_hundred_millions + delete_zero + pynutil.insert("0") + graph_millions) | (graph_hundreds + delete_hundred_millions + delete_zero + pynutil.insert("00") + graph_hundred_thousands)  
        
        #grammar for hundred millions 千億
        graph_hundred_billions = (graph_thousands + delete_hundred_millions + graph_ten_millions) | (graph_thousands + delete_hundred_millions + delete_zero + pynutil.insert("0") + graph_millions) | (graph_thousands + delete_hundred_millions + delete_zero + pynutil.insert("00") + graph_hundred_thousands)
        
        #combine all the graphs
        graph = pynini.union(graph_hundred_billions, graph_ten_billions, graph_billions, graph_hundred_millions, graph_ten_millions, graph_millions, graph_hundred_thousands, graph_ten_thousands, graph_thousands, graph_hundreds, graph_all, zero)
        
        #Formatting grammar
        #Removing leading zero (when there is a zero infront of a non-zero digit, e.g., 09,08)
        delete_leading_zeros = pynutil.delete(pynini.closure("0")) #delete "0" under closure == regex * operator
        stop_at_non_zero = pynini.difference(NEMO_DIGIT,"0") #creates a graph that accepts all input-outputs from NEMO_DGIT except "0"
        rest_of_cardinal = pynini.closure(NEMO_DIGIT) #accepts all digits that may follow
        clean_cardinal = delete_leading_zeros + stop_at_non_zero + rest_of_cardinal
        clean_cardinal = clean_cardinal | "0" #Allow the existence of a "0"
        graph =  graph @ clean_cardinal
        
        #new graph
        self.just_cardinals = graph

        #Token insertion
        optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("负", "\"-\"") + " ", 0, 1) 
        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph
        



