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

def print_template_Makefile(file_list_w, platform, sdk, backend):
    # Generate the Makefile, including all files to upload on the hyperflash
    tk = OrderedDict([])
    tk['build_layers'] = os.listdir('./application/DORY_network/src/')
    tk['layers_w'] = file_list_w
    tk['platform'] = 'GAP8'
    tk['sdk'] = sdk
    root = '/'.join(os.getcwd().split('/')[:-1])
    tmpl = Template(filename=root + f"/Templates/{backend}/Makefile_template")
    s = tmpl.render(**tk)
    if backend == 'MCU':
        save_string = './application/Makefile'
    else:
        save_string = './application/CMakeLists.txt'
    with open(save_string, "w") as f:
        f.write(s)
