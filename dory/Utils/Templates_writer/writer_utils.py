#
# template.py
# Alessio Burrello <alessio.burrello@unibo.it>
# Thorir Mar Ingolfsson <thoriri@iis.ee.ethz.ch>
#
# Copyright (C) 2019-2020 University of Bologna
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

import math
from mako.template import Template
import re
from collections import OrderedDict
import numpy as np
import sys
import os
import re

def print_file_list(x):
    # This function is used to generate a string with all input files.
    s = repr(x).replace("[", "").replace("]", "").replace("'", '"')
    return s

def print_test_vector(x, type_data):
    # Print the test vector in the c file.
    if type_data == 'uint32_t':
        try:
            np.set_printoptions(threshold=sys.maxsize,formatter={'int': lambda x: hex(np.uint8(x)) if (x < 0) else hex(np.uint8(x)), } )
        except TypeError:
            np.set_printoptions(threshold=sys.maxsize)
        s = repr([hex(a) for a in x.astype(np.uint32)]).replace("'", "").replace("array([", "").replace("]", "").replace("[", "").replace(")", "").replace(",\n      dtype=int8)", "").replace(", dtype=uint8", "").replace(",\n      dtype=uint8)", "").replace(",\n      dtype=uint8", "").replace(",\n      dtype=int8", "").replace(", dtype=int8", "").replace(", dtype=int8)", "").replace(", dtype=int8)", "").replace(", dtype=uint8)", "")

    elif type_data == 'char':
        try:
            np.set_printoptions(threshold=sys.maxsize,formatter={'int': lambda x: hex(np.uint8(x)) if (x < 0) else hex(np.uint8(x)), } )
        except TypeError:
            np.set_printoptions(threshold=sys.maxsize)
        s = repr(x.flatten()).replace("'", "").replace("array([", "").replace("]", "").replace("[", "").replace(")", "").replace(",\n      dtype=int8)", "").replace(", dtype=uint8", "").replace(",\n      dtype=uint8)", "").replace(",\n      dtype=uint8", "").replace(",\n      dtype=int8", "").replace(", dtype=int8", "").replace(", dtype=int8)", "").replace(", dtype=int8)", "").replace(", dtype=uint8)", "")

    elif type_data == 'int16_t':
        try:
            np.set_printoptions(
                threshold=sys.maxsize,
                formatter={'int': lambda x: hex(np.uint16(x)) if (
                    x < 0) else hex(np.int16(x)), }
            )
        except TypeError:
            np.set_printoptions(threshold=sys.maxsize)
        s = repr(x.flatten()).replace("array([", "").replace("]", "").replace("[", "").replace(",\n      dtype=int16)", "").replace(
            ", dtype=int16)", "").replace(", dtype=int16)", "").replace(", dtype=uint16)", "").replace(")", "")

    else:
        try:
            np.set_printoptions(
                threshold=sys.maxsize,
                formatter={'int': lambda x: hex(np.uint32(x)) if (
                    x < 0) else hex(np.int32(x)), }
            )
        except TypeError:
            np.set_printoptions(threshold=sys.maxsize)
        s = repr(x.flatten()).replace("array([", "").replace("]", "").replace("[", "").replace(
            ",\n      dtype=int32)", "").replace(", dtype=int32)", "").replace(", dtype=int32)", "").replace(", dtype=uint32)", "")
    return s
