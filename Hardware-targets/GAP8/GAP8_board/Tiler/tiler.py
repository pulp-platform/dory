     # should work even without -*-
#
# tiling.py
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

from .tiler_conv2d import Tiler_Conv2D
from .tiler_pool2d import Tiler_Pool2D
from .tiler_add import Tiler_Add


class Tiler:
    # Class to generate the Tiling of the layer.
    def __init__(self, HW_node, previous_HW_node, code_reserved_space):
        self.HW_node = HW_node
        self.previous_HW_node = previous_HW_node
        self.code_reserved_space = code_reserved_space
        self.double_buffering = 1

    def get_tiling(self, level):
        # This function is used to create the tiling of either a convolutional layer or
        # a fully connected or a pooling layer. The relu is included automatically in conv/FC.
        if 'Conv1D' in self.HW_node.name:
            return Tiler_Conv1D(self).get_tiling()
        elif 'Conv' in self.HW_node.name or  'FullyConnected' in self.HW_node.name:
            return Tiler_Conv2D(self).get_tiling(level)
        elif 'Pool' in self.HW_node.name:
            return Tiler_Pool2D(self).get_tiling(level)
        elif 'Addition' in self.HW_node.name:
            return Tiler_Add(self).get_tiling(level)
        else:
            print("Not supported Layer.")
            return None
