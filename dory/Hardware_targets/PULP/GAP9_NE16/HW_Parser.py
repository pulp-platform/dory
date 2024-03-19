# HW_Parser.py
# Alessio Burrello <alessio.burrello@unibo.it>
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

from dory.Hardware_targets.PULP.GAP9.HW_Parser import onnx_manager as onnx_manager_gap9
from dory.Hardware_targets.PULP.GAP9_NE16.HW_Pattern_rewriter import Pattern_rewriter
from dory.Hardware_targets.PULP.GAP9_NE16.Ne16_HW_node import Ne16_HW_node
from dory.Hardware_targets.PULP.GAP9_NE16.Tiler.tiler import Tiler_GAP9
import numpy as np
import os
import sys

from dory.Parsers.HW_node import HW_node
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "Backend_Kernels", "pulp-nnx", "test"))
from Ne16TestConf import Ne16TestConf
from TestClasses import IntegerType
from Ne16MemoryLayout import Ne16MemoryLayout
from pydantic import ValidationError


class onnx_manager(onnx_manager_gap9):

    def get_file_path(self):
        return "/".join(os.path.realpath(__file__).split("/")[:-1])

    def get_pattern_rewriter(self):
        return Pattern_rewriter

    def get_tiler(self):
        return Tiler_GAP9

    def valid_ne16_node(self, node, idx):
        try:
            Ne16TestConf(
                in_height=node.input_dimensions[0],
                in_width=node.input_dimensions[1],
                in_channel=node.input_channels,
                out_channel=node.output_channels,
                padding={"top": node.pads[0], "left": node.pads[1], "bottom": node.pads[2], "right": node.pads[3]},
                kernel_shape={"height": node.kernel_shape[0], "width": node.kernel_shape[1]},
                depthwise=node.group>1,
                stride={"height": node.strides[0], "width": node.strides[1]},
                in_type=IntegerType(name=f"{node.input_activation_type}{node.input_activation_bits}"),
                out_type=IntegerType(name=f"{node.output_activation_type}{node.output_activation_bits}"),
                weight_type=IntegerType(name=f"{node.weight_type}{node.weight_bits}"),
                scale_type=IntegerType(name=f"uint{node.constant_bits}"), # TODO should it be constant_type?
                bias_type=IntegerType(name=f"{node.constant_type}{node.bias_bits}"),
                has_norm_quant=True, # TODO
                has_bias=True,
                has_relu=node.min >= 0
            )
            return True, ""
        except ValidationError as e:
            msg = f"WARNING: Failed allocating node {node.name}{idx} to the NE16 accelerator. Errors:\n"
            for error in e.errors():
                msg += f" - {error['msg']}\n"
            msg += "\nNOTE: Falling back to cluster engine.\n"
            return False, msg

    def engine_coloring(self, node, idx):
        if "Conv" in node.op_type or "Convolution" in node.op_type:
            is_valid, msg = self.valid_ne16_node(node, idx)
            if is_valid:
                node.engine = "ne16"
                return
            else:
                print(msg)
        node.engine = "cluster"

    def mapping_to_HW_nodes(self):
        super().mapping_to_HW_nodes()
        print("\nPULP Backend: Assigning nodes to engines.")
        for i, node in enumerate(self.DORY_Graph):
            self.engine_coloring(node, i)
        assert all(hasattr(node, "engine") for node in self.DORY_Graph)

    def transform_nodes_to_hw_nodes(self):
        new_graph = []
        for node in self.DORY_Graph:
            if node.engine == "ne16":
                new_graph.append(Ne16_HW_node(node, self.HW_description))
            else:
                new_graph.append(HW_node(node, self.HW_description))
        self.DORY_Graph = new_graph

    def adjust_node_data_layout(self, node, node_id):
        if node.engine != "ne16":
            return super().adjust_node_data_layout(node, node_id)
        
        weights = self._get_weights_attr(node)

        # Adjust offset
        weights_offset = -(2**(node.weight_bits-1))
        weights["value"] = weights["value"] + weights_offset

        # Unroll
        ## Unroll expects layout to be "CoutCinK"
        if weights["layout"] == "CoutKCin":
            weights["value"] = np.transpose(weights["value"], (0,3,1,2))
        weights["value"] = Ne16MemoryLayout.weightEncode(weights["value"].astype(np.uint8),
                                              node.weight_bits, node.group > 1)
        weights["layout"] = "CoutCinMajKQwCinMin" # Ne16's special layout

        self.adjust_padding(node)

    def adjust_padding(self, node):
        inp_dim = node.input_dimensions
        ks = node.kernel_shape
        s = node.strides
        padding_top, padding_left, padding_bottom, padding_right = node.pads
        node.pads[2] = padding_bottom if (inp_dim[0] - ks[0] + padding_top + padding_bottom) % s[0] == 0 else 0
        node.pads[3] = padding_right if (inp_dim[1] - ks[1] + padding_left + padding_right) % s[1] == 0 else 0
