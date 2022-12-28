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

class Tiler_Conv2D():
    # Class to generate the Tiling of the layer.
    def __init__(self,tiler):
        self.__dict__ = tiler.__dict__

    def get_tiling(self, level):
        # This function generate the layer function to be included in the project for the conv2d operations (Convolutions and Fully Connected layers).
        if level == 2:
            # L3 tiling
            tiling = self.get_tiling_conv2d_L2()
            return tiling
        print("Error: Either you should be in L3-L2 tiling or L2-L1 tiling")
        os._exit(0)
        
    def get_tiling_conv2d_L2(self): 
        '''
        Function To make the tile from L2 to L1
        '''
        ###############################################
        ##### PARAMETERS INITIALIZATION ###############
        ###############################################
        L1_memory_activation = self.HW_node.HW_description["memory"]["L1"]["dimension"]
        if self.HW_node.weight_bits == 2:
            L1_memory_weights = self.HW_node.HW_description["memory"]["L1_Weights_analog"]["dimension"]
            weights_mem = self.HW_node.tiling_dimensions["L2"]["weight_memory"]
        else:
            L1_memory_weights = self.HW_node.HW_description["memory"]["L1_Weights_digital"]["dimension"]
            weights_mem = self.HW_node.tiling_dimensions["L2"]["weight_memory"] + self.HW_node.tiling_dimensions["L2"]["constants_memory"] + self.HW_node.tiling_dimensions["L2"]["bias_memory"]

        inp_dim = self.HW_node.tiling_dimensions["L2"]["input_dimensions"][1:]
        out_dim = self.HW_node.tiling_dimensions["L2"]["output_dimensions"][1:]
        out_ch = self.HW_node.tiling_dimensions["L2"]["weights_dimensions"][0]
        in_ch = self.HW_node.tiling_dimensions["L2"]["input_dimensions"][0]
        ks = self.HW_node.kernel_shape
        s = self.HW_node.strides
        g = self.HW_node.group
        p = self.HW_node.pads

        ###############################################
        ##### L2 DIMENSIONS DEFINITION: EARLY EXIT ####
        ###############################################

        in_mem = self.HW_node.tiling_dimensions["L2"]["input_activation_memory"]
        out_mem = self.HW_node.tiling_dimensions["L2"]["output_activation_memory"]
        h_in   = self.HW_node.tiling_dimensions["L2"]["input_dimensions"][1]
        h_out   = self.HW_node.tiling_dimensions["L2"]["output_dimensions"][1]
        previous_layer_tiles = 2
        if self.previous_HW_node.tiling_dimensions["L2"]["output_activation_memory"] == self.previous_HW_node.tiling_dimensions["L1"]["output_activation_memory"] and \
            self.previous_HW_node.tiling_dimensions["L2"]["input_activation_memory"] == self.previous_HW_node.tiling_dimensions["L1"]["input_activation_memory"] and \
            self.previous_HW_node.tiling_dimensions["L2"]["weight_memory"] == self.previous_HW_node.tiling_dimensions["L1"]["weight_memory"]:
            previous_layer_tiles = 1
        self.HW_node.previous_layer_tiles = previous_layer_tiles
        # return immediatly if the memory fits the L1
        if (in_mem + out_mem) <= L1_memory_activation and weights_mem <= L1_memory_weights and ((self.HW_node.weight_bits == 2 and (out_ch <= 512) and (in_ch <= 128)) or self.HW_node.weight_bits == 8) and ((out_ch <= 16)  or (inp_dim[0] > 1 and inp_dim[1] > 1)):
            return (self.HW_node.tiling_dimensions["L2"]["weights_dimensions"] , [self.HW_node.tiling_dimensions["L2"]["input_dimensions"][0], h_in, self.HW_node.tiling_dimensions["L2"]["input_dimensions"][2]] , [self.HW_node.tiling_dimensions["L2"]["weights_dimensions"][0], h_out, self.HW_node.tiling_dimensions["L2"]["output_dimensions"][2]] )
        else:
            db = 1

        ###############################################
        ##### TILING OF LAYER USING ORTOOLS ###########
        ###############################################
        ###############################################
        ##### INITIALIZATION OF THE TILING VARS #######
        ###############################################
        for tile_iteration in np.arange(3):
            parameters = pywrapcp.Solver.DefaultSolverParameters()
            solver = pywrapcp.Solver("simple_CP", parameters)
            tile_n_in =  solver.IntVar(1, in_ch, 'tile_n_in')
            tile_n_out = solver.IntVar(1, out_ch, 'tile_n_out')
            tile_h_in =  solver.IntVar(ks[0], inp_dim[0], 'tile_h_in')
            tile_w_in =  solver.IntVar(ks[1], inp_dim[1], 'tile_w_in')
            tile_h_out = solver.IntVar(1, out_dim[0], 'tile_h_out')
            tile_w_out = solver.IntVar(1, out_dim[1], 'tile_w_out')
            zero_variable = solver.IntVar(0, 0, 'zero_variable')
            ###############################################
            ##### ITERATION CONSTRAINTS #################
            ###############################################
            # we do not take into account padding, so we have to take it into account also here to be compliant with row 121
            if tile_iteration == 0 and self.HW_node.weight_bits == 8:
                solver.Add(tile_h_out == out_dim[0] - (ks[0] - 1) + (s[0] - 1))
                solver.Add(tile_w_out == out_dim[1] - (ks[1] - 1) + (s[1] - 1))
            if tile_iteration == 1 and self.HW_node.weight_bits == 8:
                solver.Add(tile_w_out == out_dim[1] - (ks[1] - 1) + (s[1] - 1))

            ###############################################
            ##### SAFETY CONSTRAINT #################
            ###############################################
            solver.Add((tile_n_in * tile_n_out * tile_h_in * tile_w_in) < (in_ch * out_ch * inp_dim[0] * inp_dim[1]))

            ###############################################
            ##### GEOMETRICAL CONSTRAINTS #################
            ###############################################
            solver.Add(0 == (tile_h_in - ks[0]) % s[0])
            solver.Add(tile_h_out * s[0] == (tile_h_in - (ks[0] - 1) + (s[0] - 1)))
            solver.Add(tile_w_out * s[1] == (tile_w_in - (ks[1] - 1) + (s[1] - 1)))

            if g > 1:
                solver.Add(tile_n_in == tile_n_out)

            if previous_layer_tiles == 1:
                solver.Add(tile_n_in == int(in_ch))
                solver.Add(tile_w_in == inp_dim[1])
                solver.Add(tile_h_in == inp_dim[0])
                
            ###############################################
            ##### CONSTRAINTS FOR BACKEND LIMITS ##########
            ###############################################
            
            if g == 1:
                solver.Add(tile_n_in == int(in_ch))
            
            if inp_dim[0] == 1 and inp_dim[1] == 1:
                solver.Add(tile_n_out <= 16)
            
            ###############################################
            ##### CONSTRAINTS FOR DIMENSION ###############
            ###############################################
            input_tile_dimension  = db * tile_n_in * tile_h_in * tile_w_in * self.HW_node.input_activation_bits // 8
            output_tile_dimension = db * tile_n_out * tile_h_out * tile_w_out * self.HW_node.output_activation_bits // 8
            if g == 1:
                weight_tile_dimension = db * (tile_n_in * tile_n_out * np.prod(ks) * self.HW_node.weight_bits // 8 // g + (1 if self.HW_node.weight_bits == 8 else 0) * tile_n_out * self.HW_node.bias_bits // 8)
            else:
                weight_tile_dimension = db * (tile_n_in * tile_n_out * np.prod(ks) * self.HW_node.weight_bits // 8 // g * 16 + (1 if self.HW_node.weight_bits == 8 else 0) * tile_n_out * self.HW_node.bias_bits // 8 * 16)
            constraint_all = input_tile_dimension + output_tile_dimension + weight_tile_dimension

            solver.Add((input_tile_dimension + output_tile_dimension) <= L1_memory_activation)
            solver.Add(weight_tile_dimension <= L1_memory_weights)
            if self.HW_node.weight_bits == 2:
                solver.Add(tile_n_out <= 512)
            
            ###############################################
            ##### HEURISTICS ADDITION #####################
            ###############################################
            obj_expr = solver.IntVar(0, 1000000000000, "obj_expr")
            heuristics = 0
            ### Maximization of Hardware Accelerator #####
            if g == 1 and self.HW_node.weight_bits == 8:
                heuristics += 1000000 * ((tile_w_out - 1) % 16) \
                            + 1000000 * ((tile_n_out - 1) % 16) \
                            + 1000000 * (tile_w_out * tile_h_out >= 16)
            elif g > 1 and self.HW_node.weight_bits == 8:
                heuristics += 1000000 * ((tile_w_out - 1) % 16) \
                            + 1000000 * (tile_w_out * tile_h_out >= 16)
            elif g == 1 and self.HW_node.weight_bits == 2:
                heuristics += 100000000 * ((tile_n_in - 1) % 128) \
                            + 100000000 * ((tile_n_out - 1) % 512)  
            elif g > 1 and self.HW_node.weight_bits == 2:
                ### TO DO: I THINK IT IS USELESS
                pass
            ####### Minimization of DMA copies #######
            if self.HW_node.weight_bits == 8:
                heuristics +=  1000000 * tile_w_out
                heuristics +=  100000 * tile_n_out
            # ####### Total Dimension of Tile ###############
            heuristics += constraint_all
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
                if tile_h_in >= inp_dim[0]:
                    tile_h_in = inp_dim[0]
                    tile_h_out = int((tile_h_in -(ks[0] - 1) + (p[0] + p[2]) + (s[0] - 1))/s[0])
                if tile_w_in >= inp_dim[1]:
                    tile_w_in = inp_dim[1]
                    tile_w_out = int((tile_w_in -(ks[1] - 1) + (p[1] + p[3]) + (s[0] - 1))/s[0]) 
                return ([tile_n_out, tile_n_in], [tile_n_in, tile_h_in, tile_w_in], [tile_n_out, tile_h_out, tile_w_out])
        print("  Conv2d ERROR: no L2-L1 tiling found of layer {} with dimensions {} / {}, input / output channels {} / {}. Exiting...".format(self.HW_node.__dict__["name"], self.HW_node.__dict__["input_dimensions"], self.HW_node.__dict__["output_dimensions"], self.HW_node.__dict__["input_channels"], self.HW_node.__dict__["output_channels"] ))
        os._exit(0)
        return None

