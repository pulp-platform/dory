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

import numpy as np
import math
import os
import sys
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import solver_parameters_pb2

class Tiler_Add_PULP():
    # Class to generate the Tiling of the layer.
    def __init__(self,tiler):
        self.__dict__ = tiler.__dict__

    def get_tiling(self, level):
        # This function generate the layer function to be included in the project for the conv2d operations (Convolutions and Fully Connected layers).
        if level == 3:
            # L3 tiling
            tiling = self.get_tiling_Add_L3()
            return tiling
        if level == 2:
            # L3 tiling
            tiling = self.get_tiling_Add_L2()
            return tiling
        print("Error: Either you should be in L3-L2 tiling or L2-L1 tiling")
        os._exit(0)

    def get_tiling_Add_L3(self):
        L2_memory = self.HW_node.HW_description["memory"]["L2"]["dimension"] - self.code_reserved_space
        # tiling for L3-L2 management
        # parameters instantiation
        ks = self.HW_node.kernel_shape
        inp_dim = self.HW_node.input_dimensions
        out_dim = self.HW_node.output_dimensions
        out_ch = self.HW_node.output_channels
        in_ch = self.HW_node.input_channels
        s = self.HW_node.strides
        p = self.HW_node.pads

        conv_overlap_h = 2 * (ks[0] // 2) + ks[0] % 2 - 1 - (s[0] - 1)

        if self.previous_HW_node.tiling_dimensions["L3"]["output_dimensions"] != self.previous_HW_node.tiling_dimensions["L3"]["output_dimensions"]:
            input_L3 = 1
        else:
            input_L3 = 0
        buffer_total = self.HW_node.input_activation_memory + self.HW_node.output_activation_memory + self.HW_node.constants_memory

        if (buffer_total <= L2_memory) and input_L3==0:
            return ([], [self.HW_node.input_channels, self.HW_node.input_dimensions[0], self.HW_node.input_dimensions[1]], [self.HW_node.output_channels, self.HW_node.output_dimensions[0], self.HW_node.output_dimensions[1]])
        print("  Add ERROR: no L3-L2 tiling supported. Exiting...")
        os._exit(0)
        return None

    def get_tiling_Add_L2(self):
        # This function generate the layer function to be included in the project for the addition operation.
        ###############################################
        ##### PARAMETERS INITIALIZATION ###############
        ###############################################
        L1_memory = self.HW_node.HW_description["memory"]["L1"]["dimension"] - self.HW_node.HW_description["HW specific parameters"]["accelerator core0 stack"] - 7 * self.HW_node.HW_description["HW specific parameters"]["accelerator core1-7 stack"]
        inp_dim = self.HW_node.tiling_dimensions["L2"]["input_dimensions"][1:]
        out_dim = self.HW_node.tiling_dimensions["L2"]["output_dimensions"][1:]
        out_ch = self.HW_node.tiling_dimensions["L2"]["output_dimensions"][0]
        in_ch = self.HW_node.tiling_dimensions["L2"]["input_dimensions"][0]
        ks = self.HW_node.kernel_shape
        s = self.HW_node.strides
        p = self.HW_node.pads

        ###############################################
        ##### L2 DIMENSIONS DEFINITION: EARLY EXIT ####
        ###############################################
        buffer_total = self.HW_node.tiling_dimensions["L2"]["constants_memory"] + self.HW_node.tiling_dimensions["L2"]["input_activation_memory"] * int(np.ceil(1 + self.HW_node.second_input_activation_bits/self.HW_node.input_activation_bits)) + self.HW_node.tiling_dimensions["L2"]["output_activation_memory"]
        # return immediatly if the memory fits the L1
        if buffer_total <= L1_memory:
            return ([], self.HW_node.tiling_dimensions["L2"]["input_dimensions"] , self.HW_node.tiling_dimensions["L2"]["output_dimensions"] )

        db = self.double_buffering

        parameters = pywrapcp.Solver.DefaultSolverParameters()
        solver = pywrapcp.Solver("simple_CP", parameters)

        assert in_ch == out_ch

        # integer positive variables.
        tile_n = in_ch
        tile_h = solver.IntVar(1, inp_dim[0], 'tile_h_in')
        tile_w = inp_dim[1]

        # CONSTRAINTS: managing of correct dimensions (no decimal h_out and any
        # type of rounding)
        input_tile_dimension  = (db * tile_n * tile_h * tile_w * self.HW_node.input_activation_bits + 7 ) // 8 # the 7 is to account for bit precision of 1, which still occupy an entire byte
        output_tile_dimension = (db * tile_n * tile_h * tile_w * self.HW_node.output_activation_bits + 7 ) // 8 # the 7 is to account for bit precision of 1, which still occupy an entire byte
        constraint_all = input_tile_dimension * int(np.ceil(1+self.HW_node.second_input_activation_bits/self.HW_node.input_activation_bits)) + output_tile_dimension
        solver.Add(constraint_all <= L1_memory)

        # objective
        obj_expr = solver.IntVar(0, 1000000000000, "obj_expr")
        solver.Add(obj_expr == constraint_all)
        objective = solver.Maximize(obj_expr, 1)

        decision_builder = solver.Phase([tile_h],
                                        solver.CHOOSE_FIRST_UNBOUND,
                                        solver.ASSIGN_MIN_VALUE)

        # Create a solution collector.
        collector = solver.LastSolutionCollector()
        # Add the decision variables.
        collector.Add(tile_h)
        # Add the objective.
        collector.AddObjective(obj_expr)

        solver.Solve(decision_builder, [objective, collector])
        if collector.SolutionCount() > 0:
            best_solution = collector.SolutionCount() - 1
            tile_h = collector.Value(best_solution, tile_h)
            return ([], [tile_n, tile_h, tile_w], [tile_n, tile_h, tile_w])
        print("  Add ERROR: no tiling found. Exiting...")
        os._exit(0)
        return None
