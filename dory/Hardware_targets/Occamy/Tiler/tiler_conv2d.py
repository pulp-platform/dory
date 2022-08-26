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
            # L2 tiling
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
        L1_memory = self.HW_node.HW_description["memory"]["L1"]["dimension"] - self.HW_node.HW_description["HW specific parameters"]["accelerator core0 stack"] - 7 * self.HW_node.HW_description["HW specific parameters"]["accelerator core1-7 stack"]
        clusters = self.HW_node.HW_description["HW specific parameters"]["clusters"]
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
        if in_ch >= clusters:
            in_mem = in_mem / clusters

        if in_ch < clusters:
            in_mem  = self.HW_node.input_activation_bits // 8 * in_ch * (inp_dim[0] + p[0] + p[2]) * (inp_dim[1] + p[1] + p[3])
            in_mem += self.HW_node.input_activation_bits // 8 * in_ch * ((p[0] + p[2]) * (inp_dim[1] + p[1] + p[3]) + (p[1] + p[3]) * inp_dim[0])
        else:
            in_mem  = self.HW_node.input_activation_bits // 8 * int(in_ch/clusters) * (inp_dim[0] + p[0] + p[2]) * (inp_dim[1] + p[1] + p[3])
            in_mem += self.HW_node.input_activation_bits // 8 * int(in_ch/clusters)  * ((p[0] + p[2]) * (inp_dim[1] + p[1] + p[3]) + (p[1] + p[3]) * inp_dim[0])
        
        out_mem = self.HW_node.tiling_dimensions["L2"]["output_activation_memory"] / clusters
        h_in   = self.HW_node.tiling_dimensions["L2"]["input_dimensions"][1]
        h_out   = self.HW_node.tiling_dimensions["L2"]["output_dimensions"][1]
        buffer_total = self.HW_node.tiling_dimensions["L2"]["weight_memory"] / clusters + self.HW_node.tiling_dimensions["L2"]["constants_memory"] / clusters + self.HW_node.tiling_dimensions["L2"]["bias_memory"] + in_mem * self.double_buffering + out_mem
        # return immediatly if the memory fits the L1  
        if buffer_total <= L1_memory:
            if in_ch >= clusters:
                return ([int(self.HW_node.tiling_dimensions["L2"]["weights_dimensions"][0] / clusters), int(self.HW_node.tiling_dimensions["L2"]["weights_dimensions"][1] / clusters)] , [int(self.HW_node.tiling_dimensions["L2"]["input_dimensions"][0] / clusters), h_in, self.HW_node.tiling_dimensions["L2"]["input_dimensions"][2]] , [int(self.HW_node.tiling_dimensions["L2"]["weights_dimensions"][0] / clusters), h_out, self.HW_node.tiling_dimensions["L2"]["output_dimensions"][2]] )
            else:
                return ([int(self.HW_node.tiling_dimensions["L2"]["weights_dimensions"][0] / clusters), self.HW_node.tiling_dimensions["L2"]["weights_dimensions"][1]] , [self.HW_node.tiling_dimensions["L2"]["input_dimensions"][0], h_in, self.HW_node.tiling_dimensions["L2"]["input_dimensions"][2]] , [int(self.HW_node.tiling_dimensions["L2"]["weights_dimensions"][0] / clusters), h_out, self.HW_node.tiling_dimensions["L2"]["output_dimensions"][2]] )
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
        tile_h_in =  solver.IntVar(ks[0], inp_dim[0], 'tile_h_in')
        tile_w_in =  solver.IntVar(ks[1], inp_dim[1], 'tile_w_in')
        tile_h_out = solver.IntVar(1, out_dim[0], 'tile_h_out')
        tile_w_out = solver.IntVar(1, out_dim[1], 'tile_w_out')
        zero_variable = solver.IntVar(0, 0, 'zero_variable')

        ###############################################
        ##### GEOMETRICAL CONSTRAINTS #################
        ###############################################
        if g == 1 or (inp_dim[0] > 32 and inp_dim[1] > 32):
            solver.Add(0 == (tile_h_in - ks[0]) % s[0])
        if g > 1:
            solver.Add(tile_n_in == tile_n_out)
        if g == 1:
            solver.Add(tile_h_out * s[0] == (tile_h_in - (ks[0] - 1) + (s[0] - 1)))
            solver.Add(tile_w_out * s[1] == (tile_w_in - (ks[1] - 1) + (s[1] - 1)))

        ###############################################
        ##### CONSTRAINTS FOR BACKEND LIMITS ##########
        ###############################################
        if g > 1:
            solver.Add(tile_h_out * s[0] == (tile_h_in - (ks[0] - 1) + ((tile_h_in % inp_dim[0]) == 0) * (p[0] + p[2]) + (s[0] - 1)))
            solver.Add(tile_w_in == inp_dim[1])
            solver.Add(tile_w_out == out_dim[1])
            solver.Add(tile_n_out % 2 == 0)
            if in_ch >= clusters:
                solver.Add(tile_n_out <= int(out_ch/clusters))
        if g == 1:
            if in_ch >= clusters and (in_ch % clusters == 0):
                solver.Add(tile_n_in == int(np.ceil(in_ch/clusters)))
            else:
                solver.Add(tile_n_in == int(in_ch))
            if clusters>1:
                solver.Add(tile_n_out <= int(out_ch/clusters))
        solver.Add(tile_n_out % int( 8 / min(self.HW_node.input_activation_bits, self.HW_node.output_activation_bits, self.HW_node.weight_bits))==0)


        ###############################################
        ##### CONSTRAINTS FOR DIMENSION ###############
        ###############################################

        input_tile_dimension  = db * tile_n_in * (tile_h_in + p[0] + p[2]) * (tile_w_in + p[1] + p[3]) * self.HW_node.input_activation_bits // 8
        input_tile_dimension += db * tile_n_in * ((p[0] + p[2]) * (tile_w_in + p[1] + p[3]) + (p[1] + p[3]) * tile_h_in) * self.HW_node.input_activation_bits // 8
        output_tile_dimension = db * tile_n_out * tile_h_out * tile_w_out * self.HW_node.output_activation_bits // 8
        if g == 1:
            weight_tile_dimension = db * in_ch * tile_n_out * np.prod(ks) * self.HW_node.weight_bits // 8
        else:
            weight_tile_dimension = db * tile_n_in * np.prod(ks) * self.HW_node.weight_bits // 8

        constants = 0
        for name in self.HW_node.constant_names:
            if name in ["l","k"]:
                constants+=1
        if constants > 0:
            constants_tile_dimension = db * tile_n_out * constants  * self.HW_node.constant_bits // 8
        else:
            constants_tile_dimension = 0

        constraint_all = input_tile_dimension + output_tile_dimension + weight_tile_dimension + constants_tile_dimension + im2col_dimension + weight_full_prec_dimension + 20 

        solver.Add(constraint_all <= L1_memory)

        ###############################################
        ##### HEURISTICS ADDITION #####################
        ###############################################
        obj_expr = solver.IntVar(0, 1000000000000, "obj_expr")
        heuristics = 0

        ####### Geometrical Shape of Tiles ############
        heuristics += 64 * 10000 * (((((h_out-zero_variable - 1) % tile_h_out)) % 8) > 4) 
        heuristics += 64 * 10000 * (((tile_h_out - 1) % 8) > 4) \
                    + 64 * 10000 * ((tile_n_out - 1) % 8) \
                    + 64 * 10000 * (tile_w_out * tile_h_out)

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

