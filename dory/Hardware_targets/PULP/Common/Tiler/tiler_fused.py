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
import numpy as np
import math
import os
import sys
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import solver_parameters_pb2
CORES = 8


class Tiler_Fused_PULP():
    # Class to generate the Tiling of the Fused layer.
    def __init__(self,tiler):
        self.__dict__ = tiler.__dict__

    def get_tiling(self, level):
        # This function generate the layer function to be included in the project for the conv2d operations (Convolutions and Fully Connected layers).
        if level == 3:
            # L3 tiling
            tiling = self.get_tiling_conv2d_L3()
            if (self.HW_node.output_channels > tiling[0][0]) and ((self.HW_node.input_dimensions[0] > tiling[1][1]) or (self.HW_node.output_dimensions[0] > tiling[2][1])):
                print("Convolution: Tiling of weights and Input/output activation from L3 not yet working. Exiting...")
                os._exit(0)
            else:
                return tiling
        if level == 2:
            # L3 tiling
            tiling = self.get_tiling_conv2d_L2()
            return tiling
        print("Error: Either you should be in L3-L2 tiling or L2-L1 tiling")
        os._exit(0)



    def get_tiling_conv2d_L3(self):

        L2_memory = self.HW_node.HW_description["memory"]["L2"]["dimension"] - self.code_reserved_space 
        # 4 iterations, adding each time a different part to be tiled, either weights, outputs, or both. Input is forced
        
        ## This is not completly the correct check. It is not always the previous node in the graph, but it could also be one other. You have to check its input node.
        if (self.previous_HW_node.tiling_dimensions["L3"]["output_dimensions"] != self.previous_HW_node.tiling_dimensions["L2"]["output_dimensions"]) and not isinstance(self.previous_HW_node.tiling_dimensions["L2"]["output_dimensions"], type(None)):
            input_L3 = 1
            self.HW_node.L3_input = 1
        else:
            input_L3 = 0
        # tiling for L3-L2 management
        buffer_total = self.HW_node.input_activation_memory + self.HW_node.output_activation_memory + self.HW_node.weight_memory + self.HW_node.bias_memory + self.HW_node.constants_memory
        if (buffer_total <= L2_memory) and input_L3==0:
            return ([self.HW_node.output_channels, self.HW_node.input_channels], [self.HW_node.input_channels, self.HW_node.input_dimensions[0], self.HW_node.input_dimensions[1]], [self.HW_node.output_channels, self.HW_node.output_dimensions[0], self.HW_node.output_dimensions[1]])
        else:
            db_W = 1
            db_O = 1
        ks1 = self.HW_node.node0.kernel_shape
        ks2 = self.HW_node.node1.kernel_shape
        inp_dim = self.HW_node.input_dimensions
        out_dim = self.HW_node.output_dimensions
        intermediate_dim = self.HW_node.node0.output_dimensions
        out_ch = self.HW_node.output_channels
        in_ch = self.HW_node.input_channels
        intermediate_ch = self.HW_node.node0.output_channels
        s1 = self.HW_node.node0.strides
        s2 = self.HW_node.node1.strides
        g1 = self.HW_node.node0.group
        g2 = self.HW_node.node1.group
        p1 = self.HW_node.node0.pads
        p2 = self.HW_node.node1.pads

        for iteration in range(0, 4):
            parameters = pywrapcp.Solver.DefaultSolverParameters()
            solver = pywrapcp.Solver("simple_CP", parameters)
            tile_n_out = solver.IntVar(1, out_ch, 'tile_n_out')
            tile_h_out = solver.IntVar(1, out_dim[0], 'tile_h_out')
            zero_variable = solver.IntVar(0, 0, 'zero_variable')
            if input_L3 == 0:
                tile_h_in = solver.IntVar(inp_dim[0], inp_dim[0], 'tile_h_in')
                tile_h_intermediate = solver.IntVar(intermediate_dim[0], intermediate_dim[0], 'tile_h_intermediate')
                db_x = 1
            else:
                tile_h_in = solver.IntVar(ks1[0], inp_dim[0], 'tile_h_in')
                tile_h_intermediate = solver.IntVar(ks2[0], intermediate_dim[0], 'tile_h_intermediate')
                solver.Add(0 == (tile_h_in - ks1[0]) % s1[0])
                solver.Add(0 == (tile_h_intermediate - ks2[0]) % s2[0])
                db_x = 2
            if iteration == 0:
                if db_x == 1:
                    db_W = 2
                    db_O = 1
                    solver.Add(tile_h_out == out_dim[0])
                else:
                    solver.Add(tile_h_out == out_dim[0])
                    solver.Add(tile_n_out == out_ch)
                    db_W = 1
                    db_O = 1
            elif iteration == 1:
                if db_x == 1:
                    db_W = 1
                    db_O = 2
                    solver.Add(tile_n_out == out_ch)
                else:
                    solver.Add(tile_n_out == out_ch)
                    db_W = 1
                    db_O = 2
            elif iteration == 2:
                if db_x == 1:
                    db_W = 2
                    db_O = 2
                else:
                    solver.Add(tile_h_out == out_dim[0])
                    db_W = 2
                    db_O = 1
            else:
                db_W = 2
                db_O = 2

            # size constraint
            input_tile_dimension  = db_x * in_ch * tile_h_in * inp_dim[1] * self.HW_node.node0.input_activation_bits // 8
            output_tile_dimension = db_O * out_ch * tile_h_out * out_dim[1] * self.HW_node.node1.output_activation_bits // 8
            weight_tile_dimension0 = db_W * intermediate_ch * (int(in_ch / g1) * np.prod(ks1) * self.HW_node.node0.weight_bits // 8 + self.HW_node.node0.bias_bits // 8 * int(self.HW_node.node0.bias_memory != 0))
            weight_tile_dimension1 = db_W * tile_n_out * (int(intermediate_ch / g2) * np.prod(ks2) * self.HW_node.node1.weight_bits // 8 + self.HW_node.node1.bias_bits // 8 * int(self.HW_node.node1.bias_memory != 0))
            weight_tile_dimension = weight_tile_dimension0 + weight_tile_dimension1
            constants_0 = 0
            for name in self.HW_node.node0.constant_names:
                if name in ["l","k"]:
                    constants_0+=1
            constants_1 = 0
            for name in self.HW_node.node1.constant_names:
                if name in ["l","k"]:
                    constants_1+=1
            constants_tile_dimension = db_W * (tile_n_out * constants_1  * self.HW_node.node1.constant_bits // 8 + intermediate_ch * constants_0 * self.HW_node.node0.constant_bits // 8) 
            constraint_all = input_tile_dimension + output_tile_dimension + weight_tile_dimension + constants_tile_dimension
            solver.Add(constraint_all <= L2_memory)

            # geometrical constraint
            if db_x == 2 and db_O == 2:   
                solver.Add(tile_h_out * s2[0] == (tile_h_intermediate - (ks2[0] - 1) + (s2[0] - 1)))
                solver.Add(tile_h_intermediate * s1[0] == (tile_h_in - (ks1[0] - 1) + (s1[0] - 1)))
            if db_x == 2:
                solver.Add(0  == (intermediate_dim[0] - zero_variable) % ((tile_h_in - ks1[0] + s1[0]) // s1[0]))
                solver.Add(0  == (out_dim[0] - zero_variable) % ((tile_h_intermediate - ks2[0] + s2[0]) // s2[0]))


            # objective              
            obj_expr = solver.IntVar(0, 100000000000000, "obj_expr")

            ### Adding L1 optimization constraint, since if you can fit a single tile in L1 you save the space for double buffering and you can execute bigger tiles
            input_tile_dimension_L1  = in_ch * tile_h_in * inp_dim[1] * self.HW_node.node0.input_activation_bits // 8
            output_tile_dimension_L1 = tile_n_out * tile_h_out * out_dim[1] * self.HW_node.node1.output_activation_bits // 8
            weight_tile_dimension_L1 = int(in_ch / g1) * intermediate_ch * np.prod(ks1) * self.HW_node.node0.weight_bits // 8 + int(intermediate_ch / g2) * tile_n_out * np.prod(ks2) * self.HW_node.node1.weight_bits // 8

            if g1 == 1:
                im2col_dimension_L1 = 2 * CORES * np.prod(ks1) * in_ch
                weight_full_prec_dimension_L1 = 0
            else:
                im2col_dimension_L1 = CORES * (ks1[0] * (in_ch + p1[0] + p1[2]) + ks1[0]) * int( 8 / min(self.HW_node.node0.input_activation_bits, self.HW_node.node0.output_activation_bits, self.HW_node.node0.weight_bits))
                weight_full_prec_dimension_L1 = 0
                if self.HW_node.node0.weight_bits != 8:
                    weight_full_prec_dimension_L1 = 32 * 8 * 8 * np.prod(ks1) * int( 8 / min(self.HW_node.node0.input_activation_bits, self.HW_node.node0.output_activation_bits, self.HW_node.node0.weight_bits))
            if g2 == 1:
                im2col_dimension_L1 += 2 * CORES * np.prod(ks1) * intermediate_ch
            else:
                im2col_dimension_L1 += CORES * (ks2[0] * (intermediate_ch + p2[0] + p2[2]) + ks2[0]) * int( 8 / min(self.HW_node.node1.input_activation_bits, self.HW_node.node1.output_activation_bits, self.HW_node.node1.weight_bits))
                if self.HW_node.node1.weight_bits != 8:
                    weight_full_prec_dimension_L1 += 32 * 8 * 8 * np.prod(ks2) * int( 8 / min(self.HW_node.node1.input_activation_bits, self.HW_node.node1.output_activation_bits, self.HW_node.node1.weight_bits))
            constants_0 = 0
            for name in self.HW_node.node0.constant_names:
                if name in ["l","k"]:
                    constants_0+=1
            constants_1 = 0
            for name in self.HW_node.node1.constant_names:
                if name in ["l","k"]:
                    constants_1+=1

            ####ADD SUPPORT BUFFERS DIMENSIONS

            constants_tile_dimension_L1 = tile_n_out * constants_1 * self.HW_node.node1.constant_bits // 8 + intermediate_ch * constants_0 * self.HW_node.node0.constant_bits // 8
            constraint_all_L1 = input_tile_dimension_L1 + output_tile_dimension_L1 + weight_tile_dimension_L1 + constants_tile_dimension_L1 + im2col_dimension_L1 + weight_full_prec_dimension_L1 + 20 
            L1_memory = self.HW_node.HW_description["memory"]["L1"]["dimension"] - self.HW_node.HW_description["HW specific parameters"]["accelerator core0 stack"] - 7 * self.HW_node.HW_description["HW specific parameters"]["accelerator core1-7 stack"]

            # objective function: 
            # 1. constraints for pulp-nn perfromance optimization
            # 2. constraints to have all tiles of same dimension
            solver.Add(obj_expr == (constraint_all 
                                    + 100000000 * (constraint_all_L1 < L1_memory)
                                    + 100000 * ((tile_h_out - 1) % 8)
                                    + 100000 * ((tile_n_out) % 4 == 0)
                                    + 1000000 * (((out_ch - tile_n_out) % tile_n_out) == 0)
                                    + 1000000 * (((out_dim[0] - tile_h_out) % tile_h_out) == 0)
                                    + 100000 * (tile_h_out == out_dim[0])
                                    + 10000 * (tile_n_out == out_ch)))

            # maximize the objective
            objective = solver.Maximize(obj_expr, 1)
            decision_builder = solver.Phase([tile_n_out, tile_h_in, tile_h_out, tile_h_intermediate],
                                            solver.CHOOSE_FIRST_UNBOUND,
                                            solver.ASSIGN_MIN_VALUE)
            # Create a solution collector.
            collector = solver.LastSolutionCollector()
            # Add the decision variables.
            collector.Add(tile_n_out)
            collector.Add(tile_h_in)
            collector.Add(tile_h_out)
            collector.Add(tile_h_intermediate)
            # Add the objective.
            collector.AddObjective(obj_expr)
            solver.Solve(decision_builder, [objective, collector])
            if collector.SolutionCount() > 0:
                best_solution = collector.SolutionCount() - 1
                tile_n_out = collector.Value(best_solution, tile_n_out)
                tile_h_in = collector.Value(best_solution, tile_h_in)
                tile_h_out = collector.Value(best_solution, tile_h_out)
                tile_h_intermediate = collector.Value(best_solution, tile_h_intermediate)
                return ([tile_n_out, intermediate_ch, in_ch], [in_ch, tile_h_in, inp_dim[1]], [out_ch, tile_h_out, out_dim[1]])
        print("  Conv2d ERROR: no L3-L2 tiling found. Exiting...")
        os._exit(0)
        return None

    def get_tiling_conv2d_L2(self): 
        '''
        Function To make the tile from L2 to L1
        '''
        ###############################################
        ##### PARAMETERS INITIALIZATION ###############
        ###############################################
        L1_memory = self.HW_node.HW_description["memory"]["L1"]["dimension"] - self.HW_node.HW_description["HW specific parameters"]["accelerator core0 stack"] - 7 * self.HW_node.HW_description["HW specific parameters"]["accelerator core1-7 stack"]
        inp_dim = self.HW_node.tiling_dimensions["L2"]["input_dimensions"][1:]
        out_dim = self.HW_node.tiling_dimensions["L2"]["output_dimensions"][1:]
        out_ch = self.HW_node.tiling_dimensions["L2"]["weights_dimensions"][0]
        intermediate_ch = self.HW_node.tiling_dimensions["L2"]["weights_dimensions"][1]
        in_ch = self.HW_node.tiling_dimensions["L2"]["input_dimensions"][0]
        ks0 = self.HW_node.node0.kernel_shape
        s0 = self.HW_node.node0.strides
        g0 = self.HW_node.node0.group
        p0 = self.HW_node.node0.pads
        ks1 = self.HW_node.node1.kernel_shape
        s1 = self.HW_node.node1.strides
        g1 = self.HW_node.node1.group
        p1 = self.HW_node.node1.pads
        intermediate_dim = [int(inp_dim[0]/s0[0]), int(inp_dim[1]/s0[1])]

        ###############################################
        ##### L2 DIMENSIONS DEFINITION: EARLY EXIT ####
        ###############################################

        if g0 == 1:
            im2col_dim = 2 * CORES * np.prod(ks0) * in_ch * self.HW_node.node0.input_activation_bits/8
            weight_full_prec_dim = 0
        else:
            im2col_dim = CORES * (ks0[0] * (inp_dim[0] + p0[0] + p0[2]) + ks0[0]) * int( 8 / min(self.HW_node.node0.input_activation_bits, self.HW_node.node0.output_activation_bits, self.HW_node.node0.weight_bits))
            weight_full_prec_dim = 8 * np.prod(ks0) * int( 8 / min(self.HW_node.node0.input_activation_bits, self.HW_node.node0.output_activation_bits, self.HW_node.node0.weight_bits))
            if self.HW_node.weight_bits == 8:
                weight_full_prec_dim = 0
        if g1 == 1:
            im2col_dim += 2 * CORES * np.prod(ks1) * in_ch * self.HW_node.node1.input_activation_bits/8
        else:
            im2col_dim += CORES * (ks1[0] * (inp_dim[0] + p1[0] + p1[2]) + ks1[0]) * int( 8 / min(self.HW_node.node1.input_activation_bits, self.HW_node.node1.output_activation_bits, self.HW_node.node1.weight_bits))
            if self.HW_node.weight_bits != 8:
                weight_full_prec_dim += 8 * np.prod(ks1) * int( 8 / min(self.HW_nod1.node0.input_activation_bits, self.HW_node.node1.output_activation_bits, self.HW_node.node1.weight_bits))
        
        in_mem = self.HW_node.tiling_dimensions["L2"]["input_activation_memory"]
        out_mem = self.HW_node.tiling_dimensions["L2"]["output_activation_memory"]
        h_in   = self.HW_node.tiling_dimensions["L2"]["input_dimensions"][1]
        h_out   = self.HW_node.tiling_dimensions["L2"]["output_dimensions"][1]
        if self.n_memory_levels > 2 and self.HW_node.tiling_dimensions["L3"]["output_dimensions"][1] > self.HW_node.tiling_dimensions["L2"]["output_dimensions"][1]:
            h_intermediate   = self.HW_node.tiling_dimensions["L2"]["output_dimensions"][1] * s1[0] + (ks1[0] - 1) - (s1[0] - 1)
            h_in   = h_intermediate * s0[0] + (ks0[0] - 1) - (s0[0] - 1)
            inp_dim[0] = h_in
            in_mem = int(self.HW_node.tiling_dimensions["L2"]["input_activation_memory"] / self.HW_node.tiling_dimensions["L2"]["input_dimensions"][1] * h_in)
        if self.n_memory_levels > 2 and  self.HW_node.tiling_dimensions["L3"]["input_dimensions"][1] > self.HW_node.tiling_dimensions["L2"]["input_dimensions"][1]:
            h_intermediate   =  int(np.floor((self.HW_node.tiling_dimensions["L2"]["input_dimensions"][1] - (ks0[0] - 1) + (s0[0] - 1)) / s0[0]))
            h_out  = int(np.floor((h_intermediate - (ks1[0] - 1) + (s1[0] - 1)) / s1[0]))
            out_dim[0] = h_out
            out_mem = int(self.HW_node.tiling_dimensions["L2"]["output_activation_memory"] / self.HW_node.tiling_dimensions["L2"]["output_dimensions"][1] * h_out)
        if "Addition" not in self.HW_node.name and "Pool" not in self.HW_node.name:
            out_mem = int(self.HW_node.tiling_dimensions["L2"]["output_activation_memory"] / self.HW_node.tiling_dimensions["L2"]["output_dimensions"][0] * self.HW_node.tiling_dimensions["L2"]["weights_dimensions"][0])
        buffer_total = self.HW_node.tiling_dimensions["L2"]["weight_memory"] + self.HW_node.tiling_dimensions["L2"]["constants_memory"] + self.HW_node.tiling_dimensions["L2"]["bias_memory"] + in_mem + out_mem + im2col_dim + weight_full_prec_dim
        # return immediatly if the memory fits the L1  
        if buffer_total <= L1_memory:
            return (self.HW_node.tiling_dimensions["L2"]["weights_dimensions"] , [self.HW_node.tiling_dimensions["L2"]["input_dimensions"][0], h_in, self.HW_node.tiling_dimensions["L2"]["input_dimensions"][2]] , [self.HW_node.tiling_dimensions["L2"]["weights_dimensions"][0], h_out, self.HW_node.tiling_dimensions["L2"]["output_dimensions"][2]] )
        else:
            db = self.double_buffering

        ###############################################
        ##### TILING OF LAYER USING ORTOOLS ###########
        ###############################################
        ###############################################
        ##### INITIALIZATION OF THE TILING VARS #######
        ###############################################
        parameters = pywrapcp.Solver.DefaultSolverParameters()
        solver = pywrapcp.Solver("simple_CP", parameters)
        tile_n_in =  solver.IntVar(1, in_ch, 'tile_n_in')
        tile_n_out = solver.IntVar(1, out_ch, 'tile_n_out')
        tile_n_intermediate = solver.IntVar(1, intermediate_ch, 'tile_n_intermediate')
        tile_h_in =  solver.IntVar(ks0[0], inp_dim[0], 'tile_h_in')
        tile_w_in =  solver.IntVar(ks0[1], inp_dim[1], 'tile_w_in')
        tile_h_intermediate =  solver.IntVar(ks1[0], intermediate_dim[0], 'tile_h_intermediate')
        tile_w_intermediate =  solver.IntVar(ks1[1], intermediate_dim[1], 'tile_w_intermediate')
        tile_h_out = solver.IntVar(1, out_dim[0], 'tile_h_out')
        tile_w_out = solver.IntVar(1, out_dim[1], 'tile_w_out')
        zero_variable = solver.IntVar(0, 0, 'zero_variable')

        ###############################################
        ##### GEOMETRICAL CONSTRAINTS #################
        ###############################################
        
        ####### LAYER 0
        if g0 == 1 or (inp_dim[0] > 32 and inp_dim[1] > 32):
            solver.Add(0 == (tile_h_in - ks0[0]) % s0[0])
        if g0 > 1:
            solver.Add(tile_n_in == tile_n_intermediate)
        if g0 == 1:
            # if a tile dimension is equal to the total input dimension in a
            # direction, padding must be counted as well. Otherwise we can
            # simply calculate "too few" outputs in one tiling iteration (???) 
            solver.Add(tile_h_intermediate * s0[0] == (tile_h_in - (ks0[0] - 1) + (s0[0] - 1) + (p0[0] + p0[2]) * (tile_h_in == inp_dim[0])))
            solver.Add(tile_w_intermediate * s0[1] == (tile_w_in - (ks0[1] - 1) + (s0[1] - 1) + (p0[1] + p0[3]) * (tile_w_in == inp_dim[1])))
        ####### LAYER 1
        if g1 == 1 or (intermediate_dim[0] > 32 and intermediate_dim[1] > 32):
            solver.Add(0 == (tile_h_intermediate - ks1[0]) % s1[0])
        if g1 > 1:
            solver.Add(tile_n_intermediate == tile_n_out)
        if g1 == 1:
            # if a tile dimension is equal to the total input dimension in a
            # direction, padding must be counted as well. Otherwise we can
            # simply calculate "too few" outputs in one tiling iteration (???) 
            solver.Add(tile_h_out * s1[0] == (tile_h_intermediate - (ks1[0] - 1) + (s1[0] - 1) + (p1[0] + p1[2]) * (tile_h_intermediate == intermediate_dim[0])))
            solver.Add(tile_w_out * s1[1] == (tile_w_intermediate - (ks1[1] - 1) + (s1[1] - 1) + (p1[1] + p1[3]) * (tile_w_intermediate == intermediate_dim[1])))

        ###############################################
        ##### CONSTRAINTS FOR BACKEND LIMITS ##########
        ###############################################
        ####### LAYER 0
        if g0 > 1:
            if inp_dim[0] <= 32 and inp_dim[1] <= 32:
                pass
                # solver.Add(tile_h_in == inp_dim[0])
                # solver.Add(tile_w_in == inp_dim[1])
                # solver.Add(tile_h_intermediate == intermediate_dim[0])
                # solver.Add(tile_w_intermediate == intermediate_dim[1])
            elif inp_dim[0] > 32 or inp_dim[1] > 32:
                solver.Add(tile_h_intermediate * s0[0] == (tile_h_in - (ks0[0] - 1) + ((tile_h_in % inp_dim[0]) == 0) * (p0[0] + p0[2]) + (s0[0] - 1)))
                solver.Add(tile_w_in == inp_dim[1])
                solver.Add(tile_w_intermediate == intermediate_dim[1])
            solver.Add(tile_n_in % int( 8 / min(self.HW_node.node0.input_activation_bits, self.HW_node.node0.output_activation_bits, self.HW_node.node0.weight_bits))==0)
        if g0 == 1:
            solver.Add(tile_n_in == int(in_ch))
        solver.Add(tile_n_intermediate % int( 8 / min(self.HW_node.node0.input_activation_bits, self.HW_node.node0.output_activation_bits, self.HW_node.node0.weight_bits))==0)
        ####### LAYER 1
        if g1 > 1:
            if intermediate_dim[0] <= 32 and intermediate_dim[1] <= 32:
                pass
                # solver.Add(tile_h_intermediate == intermediate_dim[0])
                # solver.Add(tile_w_intermediate == intermediate_dim[1])
                # solver.Add(tile_h_out == out_dim[0])
                # solver.Add(tile_w_out == out_dim[1])
            elif intermediate_dim[0] > 32 or intermediate_dim[1] > 32:
                solver.Add(tile_h_out * s1[0] == (tile_h_intermediate - (ks1[0] - 1) + ((tile_h_intermediate % intermediate_dim[0]) == 0) * (p1[0] + p1[2]) + (s1[0] - 1)))
                solver.Add(tile_w_intermediate == intermediate_dim[1])
                solver.Add(tile_w_out == out_dim[1])
            solver.Add(tile_n_intermediate % int( 8 / min(self.HW_node.node1.input_activation_bits, self.HW_node.node1.output_activation_bits, self.HW_node.node1.weight_bits))==0)
        if g1 == 1:
            solver.Add(tile_n_intermediate == int(intermediate_ch))
        solver.Add(tile_n_out % int( 8 / min(self.HW_node.node1.input_activation_bits, self.HW_node.node1.output_activation_bits, self.HW_node.node1.weight_bits))==0)

        ###############################################
        ##### CONSTRAINTS FOR DIMENSION ###############
        ###############################################

        input_tile_dimension  = db * (tile_n_in * tile_h_in * tile_w_in * self.HW_node.node0.input_activation_bits) // 8
        output_tile_dimension = db * (tile_n_out * tile_h_out * tile_w_out * self.HW_node.node1.output_activation_bits) // 8

        if g0 == 1:
            weight_tile_dimension = db * tile_n_in * tile_n_intermediate * np.prod(ks0) * self.HW_node.node0.weight_bits // 8
            im2col_dimension = 2 * CORES * np.prod(ks0) * tile_n_in
            weight_full_prec_dimension = 0
        else:
            weight_tile_dimension = (db * tile_n_in * np.prod(ks0) * self.HW_node.node0.weight_bits) // 8
            im2col_dimension = CORES * (ks0[0] * (tile_n_in + p0[0] + p0[2]) + ks0[0]) * int( 8 / min(self.HW_node.node0.input_activation_bits, self.HW_node.node0.output_activation_bits, self.HW_node.node0.weight_bits))
            weight_full_prec_dimension = 0
            if self.HW_node.weight_bits != 8:
                weight_full_prec_dimension = db * 8 * 8 * np.prod(ks0) * int( 8 / min(self.HW_node.node0.input_activation_bits, self.HW_node.node0.output_activation_bits, self.HW_node.node0.weight_bits))
        if g1 == 1:
            weight_tile_dimension += db * tile_n_intermediate * tile_n_out * np.prod(ks1) * self.HW_node.node1.weight_bits // 8
            im2col_dimension += 2 * CORES * np.prod(ks1) * tile_n_intermediate
        else:
            weight_tile_dimension += (db * tile_n_intermediate * np.prod(ks1) * self.HW_node.node1.weight_bits) // 8
            im2col_dimension += CORES * (ks1[0] * (tile_n_intermediate + p1[0] + p1[2]) + ks1[0]) * int( 8 / min(self.HW_node.node1.input_activation_bits, self.HW_node.node1.output_activation_bits, self.HW_node.node1.weight_bits))
            if self.HW_node.weight_bits != 8:
                weight_full_prec_dimension += db * 8 * 8 * np.prod(ks1) * int( 8 / min(self.HW_node.node1.input_activation_bits, self.HW_node.node1.output_activation_bits, self.HW_node.node1.weight_bits))
        
        constants_0 = 0
        for name in self.HW_node.node0.constant_names:
            if name in ["l","k"]:
                constants_0+=1
        constants_1 = 0
        for name in self.HW_node.node1.constant_names:
            if name in ["l","k"]:
                constants_1+=1
        constants_tile_dimension = db * tile_n_intermediate * constants_0  * self.HW_node.node0.constant_bits // 8 + db * tile_n_out * constants_1 * self.HW_node.node1.constant_bits // 8
        constraint_all = self.HW_node.tiling_dimensions["L2"]["bias_memory"] + input_tile_dimension + output_tile_dimension + weight_tile_dimension + constants_tile_dimension + im2col_dimension + weight_full_prec_dimension + 40 

        solver.Add(constraint_all <= L1_memory)

        ###############################################
        ##### HEURISTICS ADDITION #####################
        ###############################################
        obj_expr = solver.IntVar(0, 1000000000000, "obj_expr")
        heuristics = 0

        ####### LAYER 0
        if g0 == 1:
            ####### Geometrical Shape of Tiles ############
            heuristics +=  2000000 * ((tile_h_intermediate - 1) % 8) \
                         + 3000000 * ((tile_w_intermediate - 1) % 2) \
                         + 1000000 * ((tile_n_intermediate - 1) % 4) \
                         + 1000000 * (tile_w_intermediate * tile_h_intermediate >= 16)
            # ####### Total Dimension of Tile ###############
            heuristics += constraint_all
            ####### Maximization of Reuse of im2col #######
            heuristics += 100000 * tile_n_intermediate 
            ####### Geometrical Shape of Border Tiles #####
            heuristics += 10000 * ((intermediate_ch-zero_variable-1) % (tile_n_intermediate)) \
                        + 10000 * (((intermediate_ch-zero_variable-1) % (tile_n_intermediate)) % 4) \
                        + 20000 * (((intermediate_dim[0]-zero_variable-1) % (tile_h_intermediate)) % 8) \
                        + 30000 * (((intermediate_dim[1]-zero_variable-1) % (tile_w_intermediate)) % 2)
        elif g0 > 1:
            ####### Geometrical Shape of Tiles ############
            heuristics += 10000 * ((tile_n_intermediate > 7)) \
                        + 20000 * ((tile_n_intermediate - 1) % 16) \
                        + 10000 * ((tile_h_intermediate % 4) == 0)
            ####### Total Dimension of Tile ###############
            heuristics += constraint_all
            ####### Maximization of Reuse of im2col #######
            heuristics += 1000 * tile_w_intermediate \
                        + 1000 * tile_h_intermediate \
                        + 100 * (((intermediate_dim[0]-zero_variable-1) % (tile_h_intermediate))) \
                        + 100 * (((intermediate_dim[1]-zero_variable-1) % (tile_w_intermediate)))
            ####### Geometrical Shape of Border Tiles #####
            heuristics += 100 * (((intermediate_ch-zero_variable-1) % (tile_n_intermediate)) > 7) \
                        + 100 * (((intermediate_dim[0]-zero_variable-1) % (tile_h_intermediate)) % 4)

        ####### LAYER 1
        if g1 == 1:
            ####### Geometrical Shape of Tiles ############
            heuristics +=  2000000 * ((tile_h_out - 1) % 8) \
                         + 3000000 * ((tile_w_out - 1) % 2) \
                         + 1000000 * ((tile_n_out - 1) % 4) \
                         + 1000000 * (tile_w_out * tile_h_out >= 16)
            # ####### Total Dimension of Tile ###############
            heuristics += constraint_all
            ####### Maximization of Reuse of im2col #######
            heuristics += 100000 * tile_n_out
            ####### Geometrical Shape of Border Tiles #####
            heuristics += 10000 * ((out_ch-zero_variable-1) % (tile_n_out)) \
                        + 10000 * (((out_ch-zero_variable-1) % (tile_n_out)) % 4) \
                        + 20000 * (((out_dim[0]-zero_variable-1) % (tile_h_out)) % 8) \
                        + 30000 * (((out_dim[1]-zero_variable-1) % (tile_w_out)) % 2)
        elif g1 > 1:
            ####### Geometrical Shape of Tiles ############
            heuristics += 10000 * ((tile_n_out > 7)) \
                        + 20000 * ((tile_n_out - 1) % 16) \
                        + 10000 * ((tile_h_out % 4) == 0)
            ####### Total Dimension of Tile ###############
            heuristics += constraint_all
            ####### Maximization of Reuse of im2col #######
            heuristics += 1000 * tile_w_out \
                        + 1000 * tile_h_out \
                        + 100 * (((out_dim[0]-zero_variable-1) % (tile_h_out))) \
                        + 100 * (((out_dim[1]-zero_variable-1) % (tile_w_out)))
            ####### Geometrical Shape of Border Tiles #####
            heuristics += 100 * (((out_ch-zero_variable-1) % (tile_n_out)) > 7) \
                        + 100 * (((out_dim[0]-zero_variable-1) % (tile_h_out)) % 4)

        solver.Add(obj_expr == heuristics)
        objective = solver.Maximize(obj_expr, 1)

        decision_builder = solver.Phase([tile_n_in, tile_n_out, tile_n_intermediate, tile_h_in, tile_h_out, tile_h_intermediate, tile_w_in, tile_w_out, tile_w_intermediate],
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

            if tile_h_in >= inp_dim[0]:
                tile_h_in = inp_dim[0]
                tile_h_intermediate = int((tile_h_in -(ks0[0] - 1) + (p0[0] + p0[2]) + (s0[0] - 1))/s0[0])
                tile_h_out = int((tile_h_intermediate -(ks1[0] - 1) + (p1[0] + p1[2]) + (s1[0] - 1))/s1[0])
            if tile_w_in >= inp_dim[1]:
                tile_w_in = inp_dim[1]
                tile_w_intermediate = int((tile_w_in -(ks0[1] - 1) + (p0[1] + p0[3]) + (s0[0] - 1))/s0[0])
                tile_w_out = int((tile_w_intermediate -(ks1[1] - 1) + (p1[1] + p1[3]) + (s1[0] - 1))/s1[0])

            return ([tile_n_out, tile_n_in], [tile_n_in, tile_h_in, tile_w_in], [tile_n_out, tile_h_out, tile_w_out])
        import pdb;pdb.set_trace()
        print("  Conv2d ERROR: no L2-L1 tiling found of layer {} with dimensions {} / {}, input / output channels {} / {}. Exiting...".format(self.HW_node.__dict__["name"], self.HW_node.node0.__dict__["input_dimensions"], self.HW_node.node1.__dict__["output_dimensions"], self.HW_node.node0.__dict__["input_channels"], self.HW_node.node1.__dict__["output_channels"] ))
        os._exit(0)
        return None

