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

class Tiler_Add():
    # Class to generate the Tiling of the layer.
    def __init__(self,tiler):
        self.__dict__ = tiler.__dict__

    def get_tiling(self, level):
        # This function generate the layer function to be included in the project for the conv2d operations (Convolutions and Fully Connected layers).
        if level == 2:
            # L3 tiling
            tiling = self.get_tiling_Add_L2()
            return tiling
        print("Error: Either you should be in L3-L2 tiling or L2-L1 tiling")
        os._exit(0)
        
    def get_tiling_Add_L2(self):
        # This function generate the layer function to be included in the project for the addition operation.
        ###############################################
        ##### PARAMETERS INITIALIZATION ###############
        ###############################################
        L1_memory = self.HW_node.HW_description["memory"]["L1"]["dimension"]
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
        buffer_total = self.HW_node.tiling_dimensions["L2"]["constants_memory"] + self.HW_node.tiling_dimensions["L2"]["input_activation_memory"] * 2 + self.HW_node.tiling_dimensions["L2"]["output_activation_memory"]
        # return immediatly if the memory fits the L1 
        previous_layer_tiles = 2
        if self.previous_HW_node.tiling_dimensions["L2"]["output_activation_memory"] == self.previous_HW_node.tiling_dimensions["L1"]["output_activation_memory"] and \
            self.previous_HW_node.tiling_dimensions["L2"]["input_activation_memory"] == self.previous_HW_node.tiling_dimensions["L1"]["input_activation_memory"]:
            previous_layer_tiles = 1
        self.HW_node.previous_layer_tiles = previous_layer_tiles 
        if buffer_total <= L1_memory:
            return ([], self.HW_node.tiling_dimensions["L2"]["input_dimensions"] , self.HW_node.tiling_dimensions["L2"]["output_dimensions"] )
        else:
            db = 2
            if previous_layer_tiles == 1:
                print("Error: Add can not be tiled since previous layer was not tiled")
                os._exit(0)

        parameters = pywrapcp.Solver.DefaultSolverParameters()
        solver = pywrapcp.Solver("simple_CP", parameters)

        # integer positive variables.
        tile_n = solver.IntVar(1, in_ch, 'tile_n')
        tile_h_in =  solver.IntVar(ks[0], inp_dim[0], 'tile_h_in')
        tile_w_in =  solver.IntVar(ks[1], inp_dim[1], 'tile_w_in')
        tile_h_out = solver.IntVar(1, out_dim[0], 'tile_h_out')
        tile_w_out = solver.IntVar(1, out_dim[1], 'tile_w_out')

        # scaling is used to ensure datasize is integer
        solver.Add(tile_h_out == tile_h_in)
        solver.Add(tile_w_out == tile_w_in)            
        solver.Add(tile_n == in_ch)

        # CONSTRAINTS: managing of correct dimensions (no decimal h_out and any
        # type of rounding)
        input_tile_dimension  = (db * in_ch * tile_h_in * inp_dim[1] * self.HW_node.input_activation_bits + 7 ) // 8 # the 7 is to account for bit precision of 1, which still occupy an entire byte
        output_tile_dimension = (db * out_ch * tile_h_out * out_dim[1] * self.HW_node.output_activation_bits + 7 ) // 8 # the 7 is to account for bit precision of 1, which still occupy an entire byte
        constraint_all = input_tile_dimension * 2 + output_tile_dimension
        solver.Add(constraint_all <= L1_memory)
        # objective
        obj_expr = solver.IntVar(0, 1000000000000, "obj_expr")
        solver.Add(obj_expr == 1000000 * constraint_all
                   + 10 * tile_w_in
                   + 1 * tile_h_in
                   + 1000 * tile_n)
        objective = solver.Maximize(obj_expr, 1)

        decision_builder = solver.Phase([tile_n, tile_h_in, tile_w_in, tile_h_out, tile_w_out],
                                        solver.CHOOSE_FIRST_UNBOUND,
                                        solver.ASSIGN_MIN_VALUE)

        # Create a solution collector.
        collector = solver.LastSolutionCollector()
        # Add the decision variables.
        collector.Add(tile_n)
        collector.Add(tile_h_in)
        collector.Add(tile_w_in)
        collector.Add(tile_h_out)
        collector.Add(tile_w_out)
        # Add the objective.
        collector.AddObjective(obj_expr)

        solver.Solve(decision_builder, [objective, collector])
        if collector.SolutionCount() > 0:
            best_solution = collector.SolutionCount() - 1
            tile_n = collector.Value(best_solution, tile_n)
            tile_h_in = collector.Value(best_solution, tile_h_in)
            tile_w_in = collector.Value(best_solution, tile_w_in)
            tile_h_out = collector.Value(best_solution, tile_h_out)
            tile_w_out = collector.Value(best_solution, tile_w_out)
            return ([], [tile_n, tile_h_in, tile_w_in], [tile_n, tile_h_out, tile_w_out])
        print("  Add ERROR: no tiling found. Exiting...")
        os._exit(0)
        return None
