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

# Libraries
import os
import sys
import numpy as np
from ortools.constraint_solver import pywrapcp


class Tiler_Conv2D:
    # Class to generate the Tiling of the layer.
    def __init__(self, node, prev_node, code_reserved_space, accelerator):
        self.node = node
        self.prev_node = prev_node
        self.code_reserved_space = code_reserved_space
        self.acc = accelerator

    def get_tiling(self, level):
        # This function generate the layer function to be included in the project for the
        # conv2d operations (Convolutions and Fully Connected layers).
        if level == 3:
            # L3 tiling
            tiling = self.get_tiling_conv2d_L3()
            if (self.node.output_channels > tiling[0][0]) \
                    and ((self.node.input_dimensions[0] > tiling[1][1]) or (
                    self.node.output_dimensions[0] > tiling[2][1])):
                print("Convolution: Tiling of weights and Input/output activation from L3 not yet working. Exiting...")
                sys.exit(0)
        elif level == 2:
            # L2 tiling
            tiling = self.get_tiling_conv2d_L2()
        else:
            print("Error: Either you should be in L3-L2 tiling or L2-L1 tiling")
            sys.exit(0)

        return tiling

    def get_tiling_conv2d_L3(self):
        # TODO In the current setup, width cannot be tiled. Is this the best solution?

        L2_memory = self.node.hw_desc["memory"]["L2"]["dimension"] - self.code_reserved_space
        # 4 iterations, adding each time a different part to be tiled, either weights, outputs, or both. Input is forced

        prev_tiling = self.prev_node.tiling_dimensions
        is_first_node = self.prev_node == self.node
        # We assume that the first nodes input is always in L2
        input_in_l2 = is_first_node or prev_tiling["L3"]["output_dimensions"] == prev_tiling["L2"]["output_dimensions"]

        self.node.tiling_dimensions["L2"]["db_x"] = 1
        self.node.tiling_dimensions["L2"]["db_y"] = 1
        self.node.tiling_dimensions["L2"]["db_w"] = 1

        # tiling for L3-L2 management
        buffer_total = self.node.input_activation_memory + self.node.output_activation_memory + self.node.weight_memory + self.node.bias_memory + self.node.constants_memory
        if (buffer_total <= L2_memory) and (input_in_l2 or is_first_node):
            return ([self.node.output_channels, self.node.input_channels],
                    [self.node.input_channels, self.node.input_dimensions[0], self.node.input_dimensions[1]],
                    [self.node.output_channels, self.node.output_dimensions[0],
                     self.node.output_dimensions[1]])

        ks = self.node.kernel_shape
        in_dim = self.node.input_dimensions
        out_dim = self.node.output_dimensions
        out_ch = self.node.output_channels
        in_ch = self.node.input_channels
        s = self.node.strides
        g = self.node.group
        p = self.node.pads
        depthwise = g > 1

        db_x = 1 if input_in_l2 else 2
        self.node.tiling_dimensions["L2"]["db_x"] = db_x
        db_scheme = [(1, 1), (1, 2), (2, 1), (2, 2)]  # (db_o, db_w)

        # Skip first iteration if input is already in l2
        if input_in_l2:
            db_scheme = db_scheme[1:]

        for db_o, db_w in db_scheme:
            self.node.tiling_dimensions["L2"]["db_y"] = db_o
            self.node.tiling_dimensions["L2"]["db_w"] = db_w

            parameters = pywrapcp.Solver.DefaultSolverParameters()
            solver = pywrapcp.Solver("simple_CP", parameters)
            n_out = solver.IntConst(out_ch)
            h_out = solver.IntConst(out_dim[0])
            tile_n_out = solver.IntVar(1, out_ch, 'tile_n_out')
            tile_h_out = solver.IntVar(1, out_dim[0], 'tile_h_out')
            tile_h_in = solver.IntVar(in_dim[0] if input_in_l2 else ks[0], in_dim[0], 'tile_h_in')
            tile_n_in = tile_n_out if depthwise else in_ch

            # size constraint
            input_tile_dimension = db_x * in_ch * tile_h_in * in_dim[1] * self.node.input_activation_bits // 8
            output_tile_dimension = db_o * out_ch * tile_h_out * out_dim[1] * self.node.output_activation_bits // 8
            weight_tile_dimension = db_w * self.acc.weights_size(tile_n_out, tile_n_in, ks, self.node.weight_bits, depthwise)

            constants_tile_dimension = 0
            for name in ["l", "k"]:
                if name in self.node.constant_names:
                    constants_tile_dimension += db_w * tile_n_out * self.node.constant_bits // 8

            total_size = input_tile_dimension + output_tile_dimension + weight_tile_dimension + constants_tile_dimension

            solver.Add(total_size <= L2_memory)

            # geometrical constraint
            if db_o == 1:
                solver.Add(tile_h_out == out_dim[0])
            else:
                solver.Add(h_out % tile_h_out == 0)

            if db_w == 1:
                solver.Add(tile_n_out == out_ch)
            else:
                solver.Add(n_out % tile_n_out == 0)

            if db_x == 2 and db_o == 2:
                solver.Add(tile_h_out * s[0] == tile_h_in - (ks[0] - 1) + (s[0] - 1))

            # TODO: Why this constraint?
            if db_x == 2:
                solver.Add(h_out % ((tile_h_in - ks[0] + s[0]) // s[0]) == 0)

            # objective
            obj_expr = solver.IntVar(0, 100000000000000, "obj_expr")

            heuristics = self.acc.heuristic_l2(tile_n_out, tile_n_in, tile_h_out, total_size, ks)

            solver.Add(obj_expr == heuristics)

            # maximize the objective
            objective = solver.Maximize(obj_expr, 1)
            decision_builder = solver.Phase([tile_n_out, tile_h_in, tile_h_out],
                                            solver.CHOOSE_FIRST_UNBOUND,
                                            solver.ASSIGN_MIN_VALUE)

            # Create a solution collector.
            collector = solver.LastSolutionCollector()

            # Add the decision variables.
            collector.Add(tile_n_out)
            collector.Add(tile_h_in)
            collector.Add(tile_h_out)

            # Add the objective.
            collector.AddObjective(obj_expr)
            solver.Solve(decision_builder, [objective, collector])
            if collector.SolutionCount() > 0:
                best_solution = collector.SolutionCount() - 1
                tile_n_out = collector.Value(best_solution, tile_n_out)
                tile_h_in = collector.Value(best_solution, tile_h_in)
                tile_h_out = collector.Value(best_solution, tile_h_out)
                return [tile_n_out, in_ch], [in_ch, tile_h_in, in_dim[1]], [out_ch, tile_h_out, out_dim[1]]

        print("  Conv2d ERROR: no L3-L2 tiling found. Exiting...")
        sys.exit(0)

    def get_tiling_conv2d_L2(self):
        '''
        Function To make the tile from L2 to L1
        '''
        ###############################################
        ##### PARAMETERS INITIALIZATION ###############
        ###############################################
        L1_memory = self.node.hw_desc["memory"]["L1"]["dimension"]\
                    - self.node.hw_desc["HW specific parameters"]["accelerator core0 stack"]\
                    - 7 * self.node.hw_desc["HW specific parameters"]["accelerator core1-7 stack"]
        in_dim = self.node.tiling_dimensions["L2"]["input_dimensions"][1:]
        out_dim = self.node.tiling_dimensions["L2"]["output_dimensions"][1:]
        out_ch = self.node.tiling_dimensions["L2"]["weights_dimensions"][0]
        in_ch = self.node.tiling_dimensions["L2"]["input_dimensions"][0]
        ks = self.node.kernel_shape
        s = self.node.strides
        g = self.node.group
        p = self.node.pads
        depthwise = g > 1

        ###############################################
        ##### L2 DIMENSIONS DEFINITION: EARLY EXIT ####
        ###############################################

        # We are recalculating these variables because the output could be tiled but the input isn't or vice versa.
        in_mem = self.node.tiling_dimensions["L2"]["input_activation_memory"]
        out_mem = self.node.tiling_dimensions["L2"]["output_activation_memory"]
        h_in   = self.node.tiling_dimensions["L2"]["input_dimensions"][1]
        h_out   = self.node.tiling_dimensions["L2"]["output_dimensions"][1]
        if self.node.tiling_dimensions["L3"]["output_dimensions"][1] > self.node.tiling_dimensions["L2"]["output_dimensions"][1]:
            h_in   = self.node.tiling_dimensions["L2"]["output_dimensions"][1] * s[0] + (ks[0] - 1) - (s[0] - 1)
            in_mem = int(self.node.tiling_dimensions["L2"]["input_activation_memory"] / self.node.tiling_dimensions["L2"]["input_dimensions"][1] * h_in)
        if self.node.tiling_dimensions["L3"]["input_dimensions"][1] > self.node.tiling_dimensions["L2"]["input_dimensions"][1]:
            h_out  = int(np.floor((self.node.tiling_dimensions["L2"]["input_dimensions"][1] - (ks[0] - 1) + (s[0] - 1)) / s[0]))
            out_mem = int(self.node.tiling_dimensions["L2"]["output_activation_memory"] / self.node.tiling_dimensions["L2"]["output_dimensions"][1] * h_out)
        if "Addition" not in self.node.name and "Pool" not in self.node.name:
            out_mem = int(self.node.tiling_dimensions["L2"]["output_activation_memory"] / self.node.tiling_dimensions["L2"]["output_dimensions"][0] * self.node.tiling_dimensions["L2"]["weights_dimensions"][0])
        buffer_total = self.node.tiling_dimensions["L2"]["weight_memory"] + self.node.tiling_dimensions["L2"]["constants_memory"] + self.node.tiling_dimensions["L2"]["bias_memory"] + in_mem + out_mem

        self.node.tiling_dimensions["L1"]["db_x"] = 1
        self.node.tiling_dimensions["L1"]["db_y"] = 1
        self.node.tiling_dimensions["L1"]["db_w"] = 1

        # return immediately if the memory fits the L1
        if buffer_total <= L1_memory:
            return (self.node.tiling_dimensions["L2"]["weights_dimensions"],
                    [self.node.tiling_dimensions["L2"]["input_dimensions"][0],
                     h_in,
                     self.node.tiling_dimensions["L2"]["input_dimensions"][2]],
                    [out_ch,
                     h_out,
                     self.node.tiling_dimensions["L2"]["output_dimensions"][2]]
                    )

        self.node.tiling_dimensions["L1"]["db_x"] = 2
        self.node.tiling_dimensions["L1"]["db_y"] = 2
        self.node.tiling_dimensions["L1"]["db_w"] = 2
        db = 2

        ###############################################
        ##### TILING OF LAYER USING ORTOOLS ###########
        ###############################################

        ###############################################
        ##### INITIALIZATION OF THE TILING VARS #######
        ###############################################
        parameters = pywrapcp.Solver.DefaultSolverParameters()
        solver = pywrapcp.Solver("simple_CP", parameters)
        n_out = solver.IntConst(out_ch)
        n_in = solver.IntConst(in_ch)
        h_out = solver.IntConst(out_dim[0])
        w_out = solver.IntConst(out_dim[1])
        tile_n_in = solver.IntVar(1, in_ch, 'tile_n_in')
        tile_n_out = solver.IntVar(1, out_ch, 'tile_n_out')
        tile_h_in = solver.IntVar(ks[0], in_dim[0], 'tile_h_in')
        tile_w_in = solver.IntVar(ks[1], in_dim[1], 'tile_w_in')
        tile_h_out = solver.IntVar(1, out_dim[0], 'tile_h_out')
        tile_w_out = solver.IntVar(1, out_dim[1], 'tile_w_out')

        ###############################################
        ##### GEOMETRICAL CONSTRAINTS #################
        ###############################################
        # Spatially tile only if h_in and w_in are bigger than input buffer size
        # is_h_tiling = in_dim[0] > ne16.INPUT_BUFFER_H
        # if not is_h_tiling:
        #     solver.Add(tile_h_in == in_dim[0])
        #     solver.Add(tile_h_out == out_dim[0])
        #
        # is_w_tiling = in_dim[1] > ne16.INPUT_BUFFER_W
        # if not is_w_tiling:
        #     solver.Add(tile_w_in == in_dim[1])
        #     solver.Add(tile_w_out == out_dim[1])

        solver.Add(tile_h_out * s[0] == (tile_h_in - (ks[0] - 1) + (s[0] - 1)))
        solver.Add(tile_w_out * s[1] == (tile_w_in - (ks[1] - 1) + (s[1] - 1)))

        if depthwise:
            solver.Add(tile_n_in == tile_n_out)
        else:
            solver.Add(tile_n_in == int(in_ch))

        ###############################################
        ##### CONSTRAINTS FOR BACKEND LIMITS ##########
        ###############################################

        # None

        ###############################################
        ##### CONSTRAINTS FOR DIMENSION ###############
        ###############################################

        # TODO Current setup forces double buffering. Is there a way to set up constraints to also consider not double
        #      buffering some things? Maybe running the constraints multiple times?
        #      e.g.
        #      db_in = solver.IntVar(1, 2,  'db_in')
        #      n_in_tiles = # calculate number of tiles
        #      solver.Add(db_in == 2 if n_in_tiles > 2)
        #      -> To solve this problem, they do multiple rounds of tiling in L3 tiling
        input_tile_dimension = db * tile_n_in * tile_h_in * tile_w_in * self.node.input_activation_bits // 8
        output_tile_dimension = db * tile_n_out * tile_h_out * tile_w_out * self.node.output_activation_bits // 8
        weight_tile_dimension = db * self.acc.weights_size(tile_n_out, tile_n_in, ks, self.node.weight_bits, depthwise)

        constants_tile_dimension = 0
        for name in ["l", "k"]:
            if name in self.node.constant_names:
                constants_tile_dimension += db * tile_n_out * self.node.constant_bits // 8

        constraint_all = input_tile_dimension + output_tile_dimension + weight_tile_dimension + constants_tile_dimension

        solver.Add(constraint_all <= L1_memory)

        ###############################################
        ##### HEURISTICS ADDITION #####################
        ###############################################
        obj_expr = solver.IntVar(0, 1000000000000, "obj_expr")

        heuristics = self.acc.heuristic_l1(n_out, n_in, h_out, w_out,
                                           tile_n_out, tile_n_in, tile_h_out, tile_w_out,
                                           constraint_all, ks, modifier=1000000)

        solver.Add(obj_expr == heuristics)
        objective = solver.Maximize(obj_expr, 1)

        decision_builder = solver.Phase([tile_n_in, tile_n_out, tile_h_in, tile_h_out, tile_w_in, tile_w_out],
                                        solver.CHOOSE_FIRST_UNBOUND,
                                        solver.ASSIGN_MIN_VALUE)

        # Create a solution collector.
        collector = solver.LastSolutionCollector()

        # Add the decision variables.
        collector.Add(tile_n_in)
        collector.Add(tile_n_out)
        collector.Add(tile_h_in)
        collector.Add(tile_h_out)
        collector.Add(tile_w_in)
        collector.Add(tile_w_out)

        # Add the objective.
        collector.AddObjective(obj_expr)
        solver.Solve(decision_builder, [objective, collector])
        if collector.SolutionCount() > 0:
            best_solution = collector.SolutionCount() - 1
            tile_n_in = collector.Value(best_solution, tile_n_in)
            tile_n_out = collector.Value(best_solution, tile_n_out)
            tile_h_in = collector.Value(best_solution, tile_h_in)
            tile_h_out = collector.Value(best_solution, tile_h_out)
            tile_w_in = collector.Value(best_solution, tile_w_in)
            tile_w_out = collector.Value(best_solution, tile_w_out)
            return [tile_n_out, tile_n_in], [tile_n_in, tile_h_in, tile_w_in], [tile_n_out, tile_h_out, tile_w_out]

        print("  Conv2d ERROR: no L2-L1 tiling found. Exiting...")
        sys.exit(0)
