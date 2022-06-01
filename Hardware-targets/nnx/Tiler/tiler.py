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

# tilers for layers
from .tiler_conv2d import Tiler_Conv2D


class Tiler:
    # Class to generate the Tiling of the layer.
    def __init__(self, node, prev_node, code_reserved_space):
        self.node = node
        self.prev_node = prev_node
        self.code_reserved_space = code_reserved_space

    def get_tiling(self, level):
        # This function is used to create the tiling of either a convolutional layer or
        # a fully connected or a pooling layer. The relu is included automatically in conv/FC.
        if 'Conv' in self.node.name or 'FullyConnected' in self.node.name:
            return Tiler_Conv2D(self.node, self.prev_node, self.code_reserved_space).get_tiling(level)
        else:
            print("Not supported Layer.")
            return None
