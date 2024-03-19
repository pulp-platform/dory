# 
# tiler.py
# Alessio Burrello <alessio.burrello@unibo.it>
# Francesco Conti <f.conti@unibo.it>
# Thorir Mar Ingolfsson <thoriri@iis.ee.ethz.ch>
# Luka Macan <luka.macan@unibo.it>
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


import sys
import numpy as np
from ortools.constraint_solver import pywrapcp

from dory.Hardware_targets.PULP.Common.Tiler.tiler import Tiler_PULP
from .heuristics import heuristic_tile_shape_l2, heuristic_tile_shape_l1, heuristic_total_size_l2, heuristic_total_size_l1
from .heuristic_util import heuristic_sum


class Tiler_Conv2D_Ne16:
    # Class to generate the Tiling of the layer.
    def __init__(self, tiler: Tiler_PULP):
        self.node = tiler.HW_node
        self.prev_node = tiler.previous_HW_node
        self.code_reserved_space = tiler.code_reserved_space
        self.n_memory_levels = tiler.n_memory_levels

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
        L2_memory = self.node.HW_description["memory"]["L2"]["dimension"] - self.code_reserved_space

        prev_tiling = self.prev_node.tiling_dimensions
        is_first_node = self.prev_node == self.node
        # We assume that the first nodes input is always in L2
        input_in_l2 = is_first_node or prev_tiling["L3"]["output_dimensions"] == prev_tiling["L2"]["output_dimensions"]

        self.node.L3_input = not input_in_l2

        buffer_total = self.node.input_activation_memory + self.node.output_activation_memory + self.node.weight_memory + self.node.bias_memory + self.node.constants_memory

        # Don't tile if the whole thing fits into L2
        if (buffer_total <= L2_memory) and input_in_l2:
            self.node.tiling_dimensions["L2"]["db_x"] = 1
            self.node.tiling_dimensions["L2"]["db_y"] = 1
            self.node.tiling_dimensions["L2"]["db_w"] = 1

            if "PointwiseDepthwisePointwise" in self.node.name:
                return ([self.node.output_channels, self.node.input_channels],
                        [self.node.input_channels, self.node.input_dimensions[0], self.node.input_dimensions[1]],
                        [self.node.output_channels, self.node.output_dimensions[0],
                        self.node.output_dimensions[1]], self.node.output_channels_list[0])
            else:
                return ([self.node.output_channels, self.node.input_channels],
                        [self.node.input_channels, self.node.input_dimensions[0], self.node.input_dimensions[1]],
                        [self.node.output_channels, self.node.output_dimensions[0],
                        self.node.output_dimensions[1]])

        # L3-L2 tiling not implemented
        assert not "PointwiseDepthwisePointwise" in self.node.name

        ks = self.node.kernel_shape
        in_dim = self.node.input_dimensions
        out_dim = self.node.output_dimensions
        out_ch = self.node.output_channels
        in_ch = self.node.input_channels
        s = self.node.strides
        g = self.node.group
        depthwise = g > 1

        db_x = 1 if input_in_l2 else 2
        db_scheme = [(1, 1), (1, 2), (2, 1), (2, 2)]  # (db_y, db_w)

        # Skip first iteration if input is already in l2
        if input_in_l2:
            db_scheme = db_scheme[1:]

        for db_y, db_w in db_scheme:
            parameters = pywrapcp.Solver.DefaultSolverParameters()
            solver = pywrapcp.Solver("simple_CP", parameters)
            h_out = out_dim[0]
            w_out = out_dim[1]
            n_out = out_ch
            h_in = in_dim[0]
            w_in = in_dim[1]
            n_in = in_ch

            # optimization variables
            opt_vars = []

            def add_var_cond(min: int, max: int, name: str, cond: bool):
                if cond:
                    var = solver.IntVar(min, max, name)
                    opt_vars.append(var)
                else:
                    var = max
                return var

            tile_h_out = add_var_cond(1, h_out, 'tile_h_out', db_y > 1)
            tile_n_out = add_var_cond(1, out_ch, 'tile_n_out', db_w > 1)
            tile_h_in = add_var_cond(ks[0], h_in, 'tile_h_in', db_x > 1)

            # We are not tiling width nor input channel
            tile_w_out = w_out
            tile_w_in = w_in
            tile_n_in = tile_n_out if depthwise else n_in

            # size constraint
            input_tile_dimension = db_x * n_in * tile_h_in * w_in * (self.node.input_activation_bits // 8)
            output_tile_dimension = tile_h_out * db_y * n_out * w_out * (self.node.output_activation_bits // 8)

            if "DepthwisePointwise" in self.node.name:
                weight_tile_dimension = db_w * (self.node.calculate_weights_size(tile_n_in, tile_n_in, ks, self.node.weight_bits, dw=True) +
                                                self.node.calculate_weights_size(tile_n_out, tile_n_in, [1, 1], self.node.weight_bits, dw=False))
                constants_tile_dimension = 0
                if 'k0' in self.node.constant_names:
                    constants_tile_dimension += tile_n_in * db_w * (self.node.constant_bits // 8)
                if 'l0' in self.node.constant_names:
                    constants_tile_dimension += tile_n_in * db_w * (self.node.bias_bits // 8)
                if 'k1' in self.node.constant_names:
                    constants_tile_dimension += tile_n_out * db_w * (self.node.constant_bits // 8)
                if 'l1' in self.node.constant_names:
                    constants_tile_dimension += tile_n_out * db_w * (self.node.bias_bits // 8)
            else:
                weight_tile_dimension = db_w * self.node.calculate_weights_size(tile_n_out, tile_n_in, ks, self.node.weight_bits, depthwise)
                constants_tile_dimension = 0
                if 'k' in self.node.constant_names:
                    constants_tile_dimension += tile_n_out * db_w * (self.node.constant_bits // 8)
                if 'l' in self.node.constant_names:
                    constants_tile_dimension += tile_n_out * db_w * (self.node.bias_bits // 8)

            total_size = input_tile_dimension + output_tile_dimension + weight_tile_dimension + constants_tile_dimension

            solver.Add(total_size <= L2_memory)

            # policy constraints
            if db_y > 1:
                solver.Add(solver.IntConst(h_out) % tile_h_out == 0)

            if db_w > 1:
                solver.Add(solver.IntConst(n_out) % tile_n_out == 0)

            if db_x > 1:
                solver.Add(solver.IntConst(h_out) % ((tile_h_in - ks[0] + s[0]) // s[0]) == 0)

            # geometrical constraints
            if db_x > 1:
                solver.Add((tile_h_in - ks[0]) % s[0] == 0)

            if db_x > 1 and db_y > 1:
                solver.Add(tile_h_in == tile_h_out * s[0] + (ks[0] - s[0]))

            # objective
            obj_expr = solver.IntVar(0, 100000000000000, "obj_expr")

            heuristics = heuristic_total_size_l2(total_size, mem_size=L2_memory)
            heuristics += heuristic_tile_shape_l2((h_in, w_in, n_in), (h_out, w_out, n_out),
                                                 (tile_h_in, tile_w_in, tile_n_in), (tile_h_out, tile_w_out, tile_n_out),
                                                 ks, self.node.weight_bits, g, s)

            solver.Add(obj_expr == heuristic_sum(heuristics))

            # maximize the objective
            objective = solver.Maximize(obj_expr, 1)
            decision_builder = solver.Phase(opt_vars,
                                            solver.CHOOSE_FIRST_UNBOUND,
                                            solver.ASSIGN_MIN_VALUE)

            # Create a solution collector.
            collector = solver.LastSolutionCollector()

            # Add the decision variables.
            for var in opt_vars:
                collector.Add(var)

            # Add the objective.
            collector.AddObjective(obj_expr)
            solver.Solve(decision_builder, [objective, collector])
            if collector.SolutionCount() > 0:
                best_solution = collector.SolutionCount() - 1
                if db_y > 1:
                    tile_h_out = collector.Value(best_solution, tile_h_out)
                if db_w > 1:
                    tile_n_out = collector.Value(best_solution, tile_n_out)
                if db_x > 1:
                    tile_h_in = collector.Value(best_solution, tile_h_in)
                self.node.tiling_dimensions["L2"]["db_x"] = db_x
                self.node.tiling_dimensions["L2"]["db_y"] = db_y
                self.node.tiling_dimensions["L2"]["db_w"] = db_w
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
        L1_memory = self.node.HW_description["memory"]["L1"]["dimension"]\
                    - self.node.HW_description["HW specific parameters"]["accelerator core0 stack"]\
                    - 7 * self.node.HW_description["HW specific parameters"]["accelerator core1-7 stack"]
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
        h_in_l2 = self.node.tiling_dimensions["L2"]["input_dimensions"][1]
        h_out_l2 = self.node.tiling_dimensions["L2"]["output_dimensions"][1]
        h_in_l3 = self.node.tiling_dimensions["L3"]["input_dimensions"][1]
        h_out_l3 = self.node.tiling_dimensions["L3"]["output_dimensions"][1]
        ch_out_output = self.node.tiling_dimensions["L2"]["output_dimensions"][0]
        ch_out_weights = self.node.tiling_dimensions["L2"]["weights_dimensions"][0]

        # Fix output height if input height was not tiled
        if h_out_l3 > h_out_l2 and h_in_l3 == h_in_l2:
            h_in_l1 = h_out_l2 * s[0] + ks[0] - s[0]
            in_mem = (in_mem // h_in_l2) * h_in_l1
        else:
            h_in_l1 = h_in_l2

        # Fix input height if output height was not tiled
        if h_in_l3 > h_in_l2 and h_out_l3 == h_out_l2:
            h_out_l1 = (h_in_l2 - ks[0] + s[0]) // s[0]
            out_mem = (out_mem // h_out_l2) * h_out_l1
        else:
            h_out_l1 = h_out_l2

        # Fix if we tiled the weights output channel but not the output tiles channel
        if "Addition" not in self.node.name and "Pool" not in self.node.name:
            out_mem = int((out_mem / ch_out_output) * ch_out_weights)

        buffer_total = self.node.tiling_dimensions["L2"]["weight_memory"] + self.node.tiling_dimensions["L2"]["constants_memory"] + self.node.tiling_dimensions["L2"]["bias_memory"] + in_mem + out_mem

        if h_in_l3 > h_in_l2 or h_out_l3 > h_out_l2:
            assert h_in_l1 == h_out_l1 * s[0] + ks[0] - s[0]

        # Add intermediate buffer
        if "PointwiseDepthwisePointwise" in self.node.name:
            buffer_total += h_in_l1 * self.node.tiling_dimensions["L2"]["input_dimensions"][2] * self.node.output_channels_list[0]
            buffer_total += h_out_l1 * self.node.tiling_dimensions["L2"]["output_dimensions"][2] * self.node.output_channels_list[0]
        elif "DepthwisePointwise" in self.node.name:
            buffer_total += h_out_l1 * self.node.tiling_dimensions["L2"]["output_dimensions"][2] * self.node.tiling_dimensions["L2"]["output_dimensions"][0]

        # return immediately if the memory fits the L1
        if buffer_total <= L1_memory:
            self.node.tiling_dimensions["L1"]["db_x"] = 1
            self.node.tiling_dimensions["L1"]["db_y"] = 1
            self.node.tiling_dimensions["L1"]["db_w"] = 1

            return (self.node.tiling_dimensions["L2"]["weights_dimensions"],
                    [self.node.tiling_dimensions["L2"]["input_dimensions"][0],
                     h_in_l1,
                     self.node.tiling_dimensions["L2"]["input_dimensions"][2]],
                    [out_ch,
                     h_out_l1,
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
        h_in = h_in_l1
        w_in = in_dim[1]
        n_in = in_ch
        h_out = h_out_l1
        w_out = out_dim[1]
        n_out = out_ch
        tile_h_out = solver.IntVar(1, out_dim[0], 'tile_h_out')
        tile_w_out = solver.IntVar(1, out_dim[1], 'tile_w_out')
        tile_n_out = solver.IntVar(1, out_ch, 'tile_n_out')
        tile_h_in = solver.IntVar(ks[0], in_dim[0] + p[0] + p[2], 'tile_h_in')
        tile_w_in = solver.IntVar(ks[1], in_dim[1] + p[1] + p[3], 'tile_w_in')
        tile_n_in = solver.IntVar(1, in_ch, 'tile_n_in')
        if "PointwiseDepthwisePointwise" in self.node.name:
            tile_n_out_pw0 = solver.IntVar(1, self.node.output_channels_list[0], 'tile_n_out_pw0')
            n_in_pw1 = self.node.input_channels_list[2]

        ###############################################
        ##### GEOMETRICAL CONSTRAINTS #################
        ###############################################

        solver.Add(tile_h_in == solver.ConditionalExpression(tile_h_out < h_out, tile_h_out * s[0] + (ks[0] - 1) - (s[0] - 1), h_in))
        solver.Add(tile_w_in == solver.ConditionalExpression(tile_w_out < w_out, tile_w_out * s[1] + (ks[1] - 1) - (s[1] - 1), w_in))

        if "PointwiseDepthwisePointwise" in self.node.name:
            solver.Add(tile_n_in == n_in)
        elif "DepthwisePointwise" in self.node.name:
            pass  # no constraint on tile_n_in
        else:
            if depthwise:
                solver.Add(tile_n_in == tile_n_out)
            else:
                solver.Add(tile_n_in == n_in)

        ###############################################
        ##### CONSTRAINTS FOR DIMENSION ###############
        ###############################################

        input_tile_dimension = db * tile_n_in * tile_h_in * tile_w_in * (self.node.input_activation_bits // 8)
        output_tile_dimension = db * tile_n_out * tile_h_out * tile_w_out * (self.node.output_activation_bits // 8)
        if "PointwiseDepthwisePointwise" in self.node.name:
            weight_tile_dimension = db * (self.node.calculate_weights_size(tile_n_out_pw0, n_in, [1, 1], self.node.weight_bits, dw=False) + \
                                          self.node.calculate_weights_size(tile_n_out_pw0, tile_n_out_pw0, ks, self.node.weight_bits, dw=True) + \
                                          self.node.calculate_weights_size(tile_n_out, n_in_pw1, [1, 1], self.node.weight_bits, dw=False))
            constants_tile_dimension = 0
            if 'k0' in self.node.constant_names:
                constants_tile_dimension += db * tile_n_out_pw0 * (self.node.constant_bits // 8)
            if 'l0' in self.node.constant_names:
                constants_tile_dimension += db * tile_n_out_pw0 * (self.node.bias_bits // 8)
            if 'k1' in self.node.constant_names:
                constants_tile_dimension += db * tile_n_out_pw0 * (self.node.constant_bits // 8)
            if 'l1' in self.node.constant_names:
                constants_tile_dimension += db * tile_n_out_pw0 * (self.node.bias_bits // 8)
            if 'k2' in self.node.constant_names:
                constants_tile_dimension += db * tile_n_out * (self.node.constant_bits // 8)
            if 'l2' in self.node.constant_names:
                constants_tile_dimension += db * tile_n_out * (self.node.bias_bits // 8)
        elif "DepthwisePointwise" in self.node.name:
            weight_tile_dimension = db * (self.node.calculate_weights_size(tile_n_in, tile_n_in, ks, self.node.weight_bits, dw=True) + \
                                          self.node.calculate_weights_size(tile_n_out, n_in, [1, 1], self.node.weight_bits, dw=False))
            constants_tile_dimension = 0
            if 'k0' in self.node.constant_names:
                constants_tile_dimension += db * tile_n_in * (self.node.constant_bits // 8)
            if 'l0' in self.node.constant_names:
                constants_tile_dimension += db * tile_n_in * (self.node.bias_bits // 8)
            if 'k1' in self.node.constant_names:
                constants_tile_dimension += db * tile_n_out * (self.node.constant_bits // 8)
            if 'l1' in self.node.constant_names:
                constants_tile_dimension += db * tile_n_out * (self.node.bias_bits // 8)
        else:
            weight_tile_dimension = db * self.node.calculate_weights_size(tile_n_out, tile_n_in, ks, self.node.weight_bits, depthwise)
            constants_tile_dimension = 0
            if 'k' in self.node.constant_names:
                constants_tile_dimension += db * tile_n_out * (self.node.constant_bits // 8)
            if 'l' in self.node.constant_names:
                constants_tile_dimension += db * tile_n_out * (self.node.bias_bits // 8)

        total_size = input_tile_dimension + output_tile_dimension + weight_tile_dimension + constants_tile_dimension

        # Add intermediate buffer
        if "PointwiseDepthwisePointwise" in self.node.name:
            # Between PW0 and DW
            total_size += tile_n_out_pw0 * tile_h_in * tile_w_in * (self.node.output_activation_bits // 8)
            # Between DW and PW1
            total_size += n_in_pw1 * tile_h_out * tile_w_out * (self.node.output_activation_bits // 8)
        elif "DepthwisePointwise" in self.node.name:
            total_size += n_in * tile_h_out * tile_w_out * (self.node.output_activation_bits // 8)

        solver.Add(total_size <= L1_memory)

        ###############################################
        ##### HEURISTICS ADDITION #####################
        ###############################################
        obj_expr = solver.IntVar(0, 1000000000000, "obj_expr")

        def rem(a, b):
            """Remainder w/o 0
            Return remainder or if the remainder is 0, `b`.
            """
            return ((a - 1) % b) + 1

        if "PointwiseDepthwisePointwise" in self.node.name:
            heuristic_args = [
                    # PW0
                    ((h_in, w_in, n_in),
                     (h_in, w_in, n_in_pw1),
                     (tile_h_in, tile_w_in, n_in),
                     (tile_h_in, tile_w_in, tile_n_out_pw0),
                     [1, 1], 1, 1),
                    # DW
                    ((h_in, w_in, n_in_pw1),
                     (h_out, w_out, n_in_pw1),
                     (tile_h_in, tile_w_in, tile_n_out_pw0),
                     (tile_h_out, tile_w_out, tile_n_out_pw0),
                     ks, g, s),
                    # PW1
                    ((h_out, w_out, n_in_pw1),
                     (h_out, w_out, n_out),
                     (tile_h_out, tile_w_out, n_in_pw1),
                     (tile_h_out, tile_w_out, tile_n_out),
                     [1, 1], 1, 1)
                    ]
        elif "DepthwisePointwise" in self.node.name:
            heuristic_args = [
                    # DW
                    ((h_in, w_in, n_in),
                     (h_out, w_out, n_in),
                     (tile_h_in, tile_w_in, tile_n_in),
                     (tile_h_out, tile_w_out, tile_n_in),
                     ks, g, s),
                    # PW
                    ((h_out, w_out, n_in),
                     (h_out, w_out, n_out),
                     (tile_h_out, tile_w_out, n_in),
                     (tile_h_out, tile_w_out, tile_n_out),
                     [1, 1], 1, 1)
                    ]
        else:
            heuristic_args = [
                    ((h_in, w_in, n_in),
                     (h_out, w_out, n_out),
                     (tile_h_in, tile_w_in, tile_n_in),
                     (tile_h_out, tile_w_out, tile_n_out),
                     ks, g, s)
                    ]

        heuristics = heuristic_total_size_l1(total_size, L1_memory)
        for layer_in, layer_out, tile_in, tile_out, ks, g, s in heuristic_args:
            border_out = [rem(solver.IntConst(layer), tile)
                          for layer, tile in zip(layer_out, tile_out)]
            border_in = [rem(solver.IntConst(layer), tile)
                         for layer, tile in zip(layer_in, tile_in)]

            heuristics += heuristic_tile_shape_l1(layer_in, layer_out,
                                               tile_in, tile_out,
                                               border_in, border_out,
                                               ks, self.node.weight_bits, g, s)

        solver.Add(obj_expr == heuristic_sum(heuristics))

        objective = solver.Maximize(obj_expr, 1)
        #objective = solver.Minimize(obj_expr, 1)

        solver_variables = [tile_n_in, tile_n_out, tile_h_in, tile_h_out, tile_w_in, tile_w_out]
        if "PointwiseDepthwisePointwise" in self.node.name:
            solver_variables.append(tile_n_out_pw0)
        decision_builder = solver.Phase(solver_variables,
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
        if "PointwiseDepthwisePointwise" in self.node.name:
            collector.Add(tile_n_out_pw0)

        # Add the objective.
        collector.AddObjective(obj_expr)
        solver.Solve(decision_builder, [objective, collector])
        if collector.SolutionCount() > 0:
            best_solution = collector.SolutionCount() - 1
            tile_n_in = collector.Value(best_solution, tile_n_in)
            tile_n_out = collector.Value(best_solution, tile_n_out)
            tile_h_in = collector.Value(best_solution, tile_h_in)
            tile_h_in = tile_h_in if tile_h_in < h_in else h_in
            tile_h_out = collector.Value(best_solution, tile_h_out)
            tile_w_in = collector.Value(best_solution, tile_w_in)
            tile_w_in = tile_w_in if tile_w_in < w_in else w_in
            tile_w_out = collector.Value(best_solution, tile_w_out)
            if "PointwiseDepthwisePointwise" in self.node.name:
                tile_n_out_pw0 = collector.Value(best_solution, tile_n_out_pw0)
                return [tile_n_out, tile_n_in], [tile_n_in, tile_h_in, tile_w_in], [tile_n_out, tile_h_out, tile_w_out], tile_n_out_pw0
            return [tile_n_out, tile_n_in], [tile_n_in, tile_h_in, tile_w_in], [tile_n_out, tile_h_out, tile_w_out]

        print("  Conv2d ERROR: no L2-L1 tiling found. Exiting...")
        sys.exit(0)
