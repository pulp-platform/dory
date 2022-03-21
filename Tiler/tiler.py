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

import math
import numpy as np
import torch
import torch.nn as nn

# constraint solver for optimization
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import solver_parameters_pb2

# tilers for layers
from tiler_add import Tiler_Add
from tiler_conv1d import Tiler_Conv1D
from tiler_conv2d import Tiler_Conv2D
from tiler_pool2d import Tiler_Pool2D


# template for output
from Layer2D_templates_writer import print_template_layer
from Layer1D_templates_writer import print_template_layer_1D
from L3_templates_writer import print_template_layer_L3
from L3_templates_writer import print_pool_template_layer_L3
import logging
import os
import sys

class Tiler():
    # Class to generate the Tiling of the layer.
    def __init__(self, module, out_ch, filter_size, stride, padding, groups, x_shape, L1_buffer, L2_buffer, platform, chip, test_location, BitIn, BitW, BitOut, BitActivation, optional_type, sdk, backend, dma_parallelization, number_of_clusters):
        self.module = module
        self.out_ch = out_ch
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.x_shape = x_shape
        self.buffer_size = L1_buffer
        self.L2_buffer_size = L2_buffer
        self.platform = platform
        self.chip = chip
        self.test_location = test_location
        self.BitIn = BitIn
        self.BitW = BitW
        self.BitOut = BitOut
        self.BitActivation = BitActivation
        self.optional_type = optional_type
        self.sdk = sdk
        self.backend = backend
        self.dma_parallelization = dma_parallelization
        self.number_of_clusters = number_of_clusters

    def get_tiling(self, **kwargs):
        # This function is used to create the tiling of either a convolutional layer or a fully connected or a pooling layer.
        # The relu is included automatically in conv/FC.
        try:
            if 'Conv1D' in self.module:
                return Tiler_Conv1D(self).get_tiling(**kwargs)
            elif 'Conv' in self.module:
                return Tiler_Conv2D(self).get_tiling(**kwargs)
            elif 'Pool' in self.module:
                return Tiler_Pool2D(self).get_tiling(**kwargs)
            elif self.module is 'Add':
                return Tiler_Add(self).get_tiling(**kwargs)
            else:
                print("Not supported Layer.")
                return None
        except:
            pass
