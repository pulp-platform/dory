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
from . import _L3_template_writer
from . import _Addition_template_writer
from . import _Conv_template_writer
from . import _Pooling_template_writer
from . import _Fused_template_writer

def print_template_layer_L3(node, tmpl_dir, out_dir):
    _L3_template_writer.print_template_layer_L3(node, tmpl_dir, out_dir)

def print_template_layer(node, layer_type, tmpl_dir, out_dir, double_buffering = 2):
    if "Addition" in node.name:
        s, s_c = _Addition_template_writer.print_template_layer_Addition(node, layer_type, tmpl_dir, out_dir, double_buffering)
    elif "Pool" in node.name:
        s, s_c = _Pooling_template_writer.print_template_layer_Pooling(node, layer_type, tmpl_dir, out_dir, double_buffering)
    elif "Conv" in node.name or "FullyConnected" in node.name:
        s, s_c = _Conv_template_writer.print_template_layer_Conv(node, layer_type, tmpl_dir, out_dir, double_buffering)
    elif "Fused" in node.name:
        s, s_c = _Fused_template_writer.print_template_layer_Fused(node, layer_type, tmpl_dir, out_dir, double_buffering)
    
    return s, s_c

