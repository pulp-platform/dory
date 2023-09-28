# 
# tiler.py
# Alessio Burrello <alessio.burrello@unibo.it>
# Francesco Conti <f.conti@unibo.it>
# Thorir Mar Ingolfsson <thoriri@iis.ee.ethz.ch>
# 
# Copyright (C) 2018-2020 University of Bologna
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

from dory.Hardware_targets.PULP.Common.Tiler.tiler import Tiler_PULP
from ..Ne16_HW_node import Ne16_HW_node
from .tiler_conv2d_ne16 import Tiler_Conv2D_Ne16


class Tiler_GAP9(Tiler_PULP):
    def __init__(self, HW_node, previous_HW_node, code_reserved_space, double_buffering=2):
        super().__init__(HW_node, previous_HW_node, code_reserved_space, double_buffering)

    def get_tiling(self, level):
        if 'Conv' in self.HW_node.name and isinstance(self.HW_node, Ne16_HW_node):
            return Tiler_Conv2D_Ne16(self).get_tiling(level)
        return super().get_tiling(level)
