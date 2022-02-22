# -*- coding: future_fstrings -*-     # should work even without -*-
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

# template for output
from template import print_template_layer
from template import print_template_layer_1D
from template import print_template_layer_L3
from template import print_pool_template_layer_L3
import logging
import os
import sys

class Tiling():
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
                return self.get_tiling_conv1d(**kwargs)
            elif 'Conv' in self.module:
                return self.get_tiling_conv2d(**kwargs)
            elif 'Pool' in self.module:
                return self.get_tiling_pool2d(**kwargs)
            elif self.module is 'Add':
                return self.get_tiling_Add(**kwargs)
            else:
                print("Not supported Layer.")
                return None
        except:
            pass

##############################################
############ CONV 1D TILING ##################
##############################################

    def get_tiling_conv_1D_normal(self, fs,
                        p,
                        stride,
                        dilation,
                        in_channels,
                        out_channels,
                        x_shape,
                        y_shape,
                        BN,
                        buffer_size=44000
                        ):
        #### LIMITATION: PADDING ONLY IN FIRST TILE
        
        s = stride
        n_in = in_channels
        n_out = out_channels
        h_in = x_shape 
        # p = 0
        h_out = y_shape 
        max_tile_n_out = n_out 
        max_tile_n_in = n_in 
        min_tile_h_in = fs
        min_tile_h_out = 1  
        # this is to renormalize all costs
        max_obj_value = sys.maxsize
        # constraints
        input_dim = 8 * n_in * h_in
        output_dim = 8 * n_out * h_out
        weight_dim = 8 * n_in * n_out * fs
        im2col_dim = 8 * 2 * 8 * fs * n_in
        bn_dim = self.BitActivation * n_out * 2
        buffer_total = input_dim + output_dim + weight_dim + im2col_dim + bn_dim
        if BN == 0:
            buffer_total -= bn_dim
        if buffer_total <= buffer_size * 8:
            ### MALE MALE MALE
            obj_expr = 58+(h_out/2)*(119+(n_out/4)*((14*n_in*fs/4)+159))
            print(f'tile in: {n_in} x {h_in}, tile out: {n_out} x {h_out}' )
            return (n_in, n_out, h_in, h_out, obj_expr, buffer_total/8)
        else:
            db = 2
        parameters = pywrapcp.Solver.DefaultSolverParameters()
        solver = pywrapcp.Solver("simple_CP", parameters)
        # ottimizzato channel first per kernel HWC
        tile_n_in = solver.IntVar(1, max_tile_n_in, 'tile_n_out')
        tile_n_out = solver.IntVar(1, max_tile_n_out, 'tile_n_out')
        tile_h_in = solver.IntVar(min_tile_h_in, h_in, 'tile_h_in')
        tile_h_out = solver.IntVar(min_tile_h_out, h_out, 'tile_h_out')
        zero_variable = solver.IntVar(0, 0, 'zero variable')
        h_out_intvar = solver.IntVar(min_tile_h_out,h_out,'h_out_intvar')
        solver.Add(h_out_intvar == h_out)
        if ((fs - 1)*dilation+1)*2 <= h_in:
            solver.Add(0 == (tile_h_in - ((fs - 1)*dilation+1)) % s)
            solver.Add(tile_h_out * s == (tile_h_in - (fs - 1)*dilation + (s - 1)))
            # padding added
            solver.Add(solver.Max((h_in - tile_h_in - (tile_h_in - (fs - 1)*dilation - p)), 0) % (tile_h_in - (fs - 1)*dilation + 1) + abs(solver.Min(solver.Max((h_in - tile_h_in - (tile_h_in - (fs - 1)*dilation - p)), 0) % (tile_h_in - (fs - 1)*dilation), 1) - 1) * ((fs - 1)*dilation+1) >= ((fs - 1)*dilation+1))
            #TO MAKE SURE TILING doesn't fail for dilation

            solver.Add(h_in >= s*(tile_h_out*(h_out_intvar//tile_h_out)-1)-p+dilation*(fs-1)+1)
        else:
            solver.Add(h_in == tile_h_in )
            solver.Add(h_out == tile_h_out )
        solver.Add(tile_n_in == n_in)
        constr_in = db * 8 * tile_n_in * tile_h_in #* tile_w_in
        constr_out = db * 8 * tile_n_out * tile_h_out #* tile_w_out
        constr_weight = db * 8 * tile_n_in * tile_n_out * fs
        constr_im2col = 2 * 8 * 8 * fs * tile_n_in
        constr_bn = self.BitActivation * n_out * 2
        constraint_all = constr_in + constr_out + constr_weight + constr_bn + constr_im2col + 20*32 # 20 are the + 4 added between buffers
        if BN == 0:
            constraint_all -= constr_bn
        solver.Add(constraint_all <= buffer_size * 8)
        # objective
        obj_expr = solver.IntVar(0, max_obj_value, "obj_expr")
        n_tiles_h = solver.IntVar(1, h_in, "n_tiles_h")
        leftover_tile_h_out = solver.IntVar(0, h_out, "leftover_tile_h_out")
        n_tiles_nout = solver.IntVar(1, n_out, "n_tiles_nout")
        leftover_tile_nout = solver.IntVar(0, n_out, "leftover_tile_nout")

        solver.Add(n_tiles_h == (h_out + tile_h_out - 1) // tile_h_out)
        solver.Add(leftover_tile_h_out == (h_out + zero_variable) % tile_h_out)
        solver.Add(n_tiles_nout  == (n_out + tile_n_out - 1) // tile_n_out)
        solver.Add(leftover_tile_nout == (n_out + zero_variable) % tile_n_out)
        solver.Add(obj_expr == (64 * 10000 * tile_n_out
                    + constraint_all
                    + 64 * 1000000 * ((tile_h_out - 1) % 16)
                    + 64 * 1000000 * ((tile_n_out - 1) % 4) 
                    + 64 * 10000 * (leftover_tile_nout)
                    + 64 * 10000 * (leftover_tile_nout % 4)
                    + 64 * 10000 * (leftover_tile_h_out  % 16)))
        
        objective = solver.Maximize(obj_expr, 1)

        decision_builder = solver.Phase([tile_n_in, tile_n_out, tile_h_in, tile_h_out, n_tiles_h, leftover_tile_h_out, n_tiles_nout, leftover_tile_nout],
                                        solver.CHOOSE_FIRST_UNBOUND,
                                        solver.ASSIGN_MIN_VALUE)
        
        # Create a solution collector.
        collector = solver.LastSolutionCollector()
        # Add the decision variables.
        collector.Add(tile_n_in)
        collector.Add(tile_n_out)
        collector.Add(tile_h_in)
        collector.Add(tile_h_out)
        collector.Add(n_tiles_h)
        collector.Add(leftover_tile_h_out)
        collector.Add(n_tiles_nout)
        collector.Add(leftover_tile_nout)
        collector.Add(obj_expr)
        collector.Add(constraint_all)
        # Add the objective.
        collector.AddObjective(obj_expr)
        solver.Solve(decision_builder, [objective, collector])
        if collector.SolutionCount() > 0:
            best_solution = collector.SolutionCount() - 1
            tile_n_in = collector.Value(best_solution, tile_n_in)
            tile_n_out = collector.Value(best_solution, tile_n_out)
            tile_h_in = collector.Value(best_solution, tile_h_in)
            tile_h_out = collector.Value(best_solution, tile_h_out)
            n_tiles_h = collector.Value(best_solution, n_tiles_h)
            leftover_tile_h_out = collector.Value(best_solution, leftover_tile_h_out)
            n_tiles_nout = collector.Value(best_solution, n_tiles_nout)
            leftover_tile_nout = collector.Value(best_solution, leftover_tile_nout)
            obj_expr = collector.Value(best_solution, obj_expr)
            memory = collector.Value(best_solution, constraint_all)
            if tile_h_in >= h_in:
                tile_h_in = h_in
                tile_h_out = int((tile_h_in -(fs - 1)*dilation + (2*p) + (s - 1))/s)
            MAC = tile_n_out*tile_n_in*tile_h_out*fs
            cycles_im2col = (61 + 32 * fs) * ((32 * fs) > ((fs * tile_n_in * 2) // 8) ) + (29 + (fs * tile_n_in * 2) // 8) * ((32 * fs) < ((fs * tile_n_in * 2) // 8) )
            cycles_im2col_leftover = (38 + 16 * fs) * ((16 * fs) > ((fs * tile_n_in) // 8) ) + (29 + (fs * tile_n_in) // 8) * ((16 * fs) < ((fs * tile_n_in) // 8) )
            n_4_2 = ((tile_h_out // 8 + ((tile_h_out % 8) > 0)) // 2)
            n_4_1 = ((tile_h_out // 8 + ((tile_h_out % 8) > 0)) % 2)
            cycles_chout_4_2 = (tile_n_out // 4) * (159 + 14 * ((tile_n_in*fs) // 4) + 18 * ((tile_n_in*fs) % 4))
            cycles_leftover_chout_4_2 =  (tile_n_out % 4) * (52 + ((tile_n_in*fs) // 4) * 5 + 8 * ((tile_n_in*fs) % 4))
            cycles_chout_4_1 = (tile_n_out // 4) * (82 + ((tile_n_in*fs) //4) * 9 + 12 * ((tile_n_in*fs) % 4))
            cycles_leftover_chout_4_1 = (tile_n_out % 4) * (28 + 3 * ((tile_n_in*fs) // 4) + 5 * ((tile_n_in*fs) % 4))
            instr = (100 + n_4_2 * (27 + cycles_im2col + (127 + cycles_chout_4_2 + cycles_leftover_chout_4_2)) + n_4_1 * (20 + cycles_im2col_leftover + (94 + cycles_chout_4_1 + cycles_leftover_chout_4_1)))
            constr1 = (MAC * 1000) // instr 
            ## left over h
            MAC = leftover_tile_h_out*tile_n_in*tile_n_out*fs
            cycles_im2col = (61 + 32 * fs) * ((32 * fs) > ((fs * tile_n_in * 2) // 8) ) + (29 + (fs * tile_n_in * 2) // 8) * ((32 * fs) < ((fs * tile_n_in * 2) // 8) )
            cycles_im2col_leftover = (38 + 16 * fs) * ((16 * fs) > ((fs * tile_n_in) // 8) ) + (29 + (fs * tile_n_in) // 8) * ((16 * fs) < ((fs * tile_n_in) // 8) )
            n_4_2 = ((leftover_tile_h_out // 8 + ((leftover_tile_h_out % 8) > 0)) // 2)
            n_4_1 = ((leftover_tile_h_out // 8 + ((leftover_tile_h_out % 8) > 0)) % 2)
            cycles_chout_4_2 = (tile_n_out // 4) * (159 + 14 * ((tile_n_in*fs) // 4) + 18 * ((tile_n_in*fs) % 4))
            cycles_leftover_chout_4_2 =  (tile_n_out % 4) * (52 + ((tile_n_in*fs) // 4) * 5 + 8 * ((tile_n_in*fs) % 4))
            cycles_chout_4_1 = (tile_n_out // 4) * (82 + ((tile_n_in*fs) //4) * 9 + 12 * ((tile_n_in*fs) % 4))
            cycles_leftover_chout_4_1 = (tile_n_out % 4) * (28 + 3 * ((tile_n_in*fs) // 4) + 5 * ((tile_n_in*fs) % 4))
            instr = (100 + n_4_2 * (27 + cycles_im2col + (127 + cycles_chout_4_2 + cycles_leftover_chout_4_2)) + n_4_1 * (20 + cycles_im2col_leftover + (94 + cycles_chout_4_1 + cycles_leftover_chout_4_1)))
            constr2 = (MAC * 1000) // instr
            ## left over n
            MAC = leftover_tile_nout*tile_n_in*tile_h_out*fs
            cycles_im2col = (61 + 32 * fs) * ((32 * fs) > ((fs * tile_n_in * 2) // 8) ) + (29 + (fs * tile_n_in * 2) // 8) * ((32 * fs) < ((fs * tile_n_in * 2) // 8) )
            cycles_im2col_leftover = (38 + 16 * fs) * ((16 * fs) > ((fs * tile_n_in) // 8) ) + (29 + (fs * tile_n_in) // 8) * ((16 * fs) < ((fs * tile_n_in) // 8) )
            n_4_2 = ((tile_h_out // 8 + ((tile_h_out % 8) > 0)) // 2)
            n_4_1 = ((tile_h_out // 8 + ((tile_h_out % 8) > 0)) % 2)
            cycles_chout_4_2 = (leftover_tile_nout // 4) * (159 + 14 * ((tile_n_in*fs) // 4) + 18 * ((tile_n_in*fs) % 4))
            cycles_leftover_chout_4_2 =  (leftover_tile_nout % 4) * (52 + ((tile_n_in*fs) // 4) * 5 + 8 * ((tile_n_in*fs) % 4))
            cycles_chout_4_1 = (leftover_tile_nout // 4) * (82 + ((tile_n_in*fs) //4) * 9 + 12 * ((tile_n_in*fs) % 4))
            cycles_leftover_chout_4_1 = (leftover_tile_nout % 4) * (28 + 3 * ((tile_n_in*fs) // 4) + 5 * ((tile_n_in*fs) % 4))
            instr = (100 + n_4_2 * (27 + cycles_im2col + (127 + cycles_chout_4_2 + cycles_leftover_chout_4_2)) + n_4_1 * (20 + cycles_im2col_leftover + (94 + cycles_chout_4_1 + cycles_leftover_chout_4_1)))
            constr3 = (MAC * 1000) // instr
            ## left over all
            MAC = leftover_tile_nout*tile_n_in*leftover_tile_h_out*fs
            cycles_im2col = (61 + 32 * fs) * ((32 * fs) > ((fs * tile_n_in * 2) // 8) ) + (29 + (fs * tile_n_in * 2) // 8) * ((32 * fs) < ((fs * tile_n_in * 2) // 8) )
            cycles_im2col_leftover = (38 + 16 * fs) * ((16 * fs) > ((fs * tile_n_in) // 8) ) + (29 + (fs * tile_n_in) // 8) * ((16 * fs) < ((fs * tile_n_in) // 8) )
            n_4_2 = ((leftover_tile_h_out // 8 + ((leftover_tile_h_out % 8) > 0)) // 2)
            n_4_1 = ((leftover_tile_h_out // 8 + ((leftover_tile_h_out % 8) > 0)) % 2)
            cycles_chout_4_2 = (leftover_tile_nout // 4) * (159 + 14 * ((tile_n_in*fs) // 4) + 18 * ((tile_n_in*fs) % 4))
            cycles_leftover_chout_4_2 =  (leftover_tile_nout % 4) * (52 + ((tile_n_in*fs) // 4) * 5 + 8 * ((tile_n_in*fs) % 4))
            cycles_chout_4_1 = (leftover_tile_nout // 4) * (82 + ((tile_n_in*fs) //4) * 9 + 12 * ((tile_n_in*fs) % 4))
            cycles_leftover_chout_4_1 = (leftover_tile_nout % 4) * (28 + 3 * ((tile_n_in*fs) // 4) + 5 * ((tile_n_in*fs) % 4))
            instr = (100 + n_4_2 * (27 + cycles_im2col + (127 + cycles_chout_4_2 + cycles_leftover_chout_4_2)) + n_4_1 * (20 + cycles_im2col_leftover + (94 + cycles_chout_4_1 + cycles_leftover_chout_4_1)))
            constr4 = (MAC * 1000) // instr
            constr = (constr1*(n_tiles_h-(leftover_tile_h_out>0))*(n_tiles_nout-(leftover_tile_nout>0)) + constr2*(n_tiles_nout-(leftover_tile_nout>0))*(leftover_tile_h_out>0) + constr3*(n_tiles_h-(leftover_tile_h_out>0))*(leftover_tile_nout>0) + constr4)//((n_tiles_h-(leftover_tile_h_out>0))*(n_tiles_nout-(leftover_tile_nout>0)) + (n_tiles_nout-(leftover_tile_nout>0))*(leftover_tile_h_out>0) + (n_tiles_h-(leftover_tile_h_out>0))*(leftover_tile_nout>0) + 1*(leftover_tile_nout>0)*(leftover_tile_h_out>0))
            print(f'Conv normal MAC/cycle simulated: {(constr/1000)}')
            return (tile_n_in, tile_n_out, tile_h_in, tile_h_out, (constr/1000), memory/8)
        return None


    def get_tiling_conv_1D_nodilation(self,
                                   fs,
                                   p,
                                   stride,
                                   dilation,
                                   in_channels,
                                   out_channels,
                                   x_shape,
                                   y_shape,
                                   BN,
                                   buffer_size=44000
                                   ):
        s = stride
        n_in = in_channels
        n_out = out_channels
        h_in = x_shape 
        # p = 0
        h_out = y_shape 
        max_tile_n_out = n_out 
        max_tile_n_in = n_in 
        min_tile_h_in = fs
        min_tile_h_out = 1  
        # this is to renormalize all costs
        max_obj_value = 10000000000000
        # constraints
        input_dim = 8 * n_in * h_in
        output_dim = 8 * n_out * h_out
        weight_dim = 8 * n_in * n_out * fs
        bn_dim = self.BitActivation * n_out * 2
        buffer_total = input_dim + output_dim + weight_dim + bn_dim
        if BN == 0:
            buffer_total -= bn_dim
        if buffer_total <= buffer_size * 8:
            obj_expr = 58+(h_out/2)*(119+(n_out/4)*((14*n_in*fs/4)+159))
            print(f'tile in: {n_in} x {h_in}, tile out: {n_out} x {h_out}' )
            return (n_in, n_out, h_in, h_out, obj_expr, buffer_total/8)
        else:
            db = 2
        parameters = pywrapcp.Solver.DefaultSolverParameters()
        solver = pywrapcp.Solver("simple_CP", parameters)
        # ottimizzato channel first per kernel HWC
        tile_n_in = solver.IntVar(1, max_tile_n_in, 'tile_n_out')
        tile_n_out = solver.IntVar(1, max_tile_n_out, 'tile_n_out')
        tile_h_in = solver.IntVar(min_tile_h_in, h_in, 'tile_h_in')
        tile_h_out = solver.IntVar(min_tile_h_out, h_out, 'tile_h_out')
        zero_variable = solver.IntVar(0, 0, 'zero variable')
      
        solver.Add(0 == (tile_h_in - ((fs - 1)*dilation+1)) % s)
        solver.Add(tile_h_out * s == (tile_h_in - (fs - 1)*dilation + (s - 1)))
        solver.Add(tile_n_in == n_in)
        # padding added
        solver.Add(solver.Max((h_in - tile_h_in - (tile_h_in - fs + 1 - p)), 0) % (tile_h_in - fs + 1) + abs(solver.Min(
            solver.Max((h_in - tile_h_in - (tile_h_in - fs + 1 - p)), 0) % (tile_h_in - fs + 1), 1) - 1) * fs >= fs)
        constr_in = db * 8 * tile_n_in * tile_h_in #* tile_w_in
        constr_out = db * 8 * tile_n_out * tile_h_out #* tile_w_out
        constr_weight = db * 8 * tile_n_in * tile_n_out * fs
        constr_bn = self.BitActivation * n_out * 2
        constraint_all = constr_in + constr_out + constr_weight + constr_bn + 20*32 # 20 are the + 4 added between buffers
        if BN == 0:
            constraint_all -= constr_bn
        solver.Add(constraint_all <= buffer_size * 8)
        # objective
        obj_expr = solver.IntVar(0, max_obj_value, "obj_expr")
        n_tiles_h = solver.IntVar(1, h_in, "n_tiles_h")
        leftover_tile_h_out = solver.IntVar(0, h_out, "leftover_tile_h_out")
        n_tiles_nout = solver.IntVar(1, n_out, "n_tiles_nout")
        leftover_tile_nout = solver.IntVar(0, n_out, "leftover_tile_nout")
        solver.Add(n_tiles_h == (h_out + tile_h_out - 1) // tile_h_out)
        solver.Add(leftover_tile_h_out == (h_out + zero_variable) % tile_h_out)
        solver.Add(n_tiles_nout  == (n_out + tile_n_out - 1) // tile_n_out)
        solver.Add(leftover_tile_nout == (n_out + zero_variable) % tile_n_out)
        ## principal kernel
        solver.Add(obj_expr == (64 * 10000 * tile_n_out
                    + constraint_all
                    + 64 * 1000000 * ((tile_h_out - 1) % 16)
                    + 64 * 1000000 * ((tile_n_out - 1) % 4) 
                    + 64 * 10000 * (leftover_tile_nout)
                    + 64 * 10000 * (leftover_tile_nout % 4)
                    + 64 * 10000 * (leftover_tile_h_out  % 16)))
        objective = solver.Maximize(obj_expr, 1)

        decision_builder = solver.Phase([tile_n_in, tile_n_out, tile_h_in, tile_h_out, n_tiles_h, leftover_tile_h_out, n_tiles_nout, leftover_tile_nout],
                                        solver.CHOOSE_FIRST_UNBOUND,
                                        solver.ASSIGN_MIN_VALUE)
        
        # Create a solution collector.
        collector = solver.LastSolutionCollector()
        # Add the decision variables.
        collector.Add(tile_n_in)
        collector.Add(tile_n_out)
        collector.Add(tile_h_in)
        collector.Add(tile_h_out)
        collector.Add(n_tiles_h)
        collector.Add(leftover_tile_h_out)
        collector.Add(n_tiles_nout)
        collector.Add(leftover_tile_nout)
        collector.Add(obj_expr)
        collector.Add(constraint_all)
        # Add the objective.
        collector.AddObjective(obj_expr)
        solver.Solve(decision_builder, [objective, collector])
        if collector.SolutionCount() > 0:
            best_solution = collector.SolutionCount() - 1
            tile_n_in = collector.Value(best_solution, tile_n_in)
            tile_n_out = collector.Value(best_solution, tile_n_out)
            tile_h_in = collector.Value(best_solution, tile_h_in)
            tile_h_out = collector.Value(best_solution, tile_h_out)
            n_tiles_h = collector.Value(best_solution, n_tiles_h)
            leftover_tile_h_out = collector.Value(best_solution, leftover_tile_h_out)
            n_tiles_nout = collector.Value(best_solution, n_tiles_nout)
            leftover_tile_nout = collector.Value(best_solution, leftover_tile_nout)
            obj_expr = collector.Value(best_solution, obj_expr)
            memory = collector.Value(best_solution, constraint_all)
            if tile_h_in >= h_in:
                tile_h_in = h_in
                tile_h_out = int((tile_h_in -(fs - 1)*dilation + (2*p) + (s - 1))/s)
            MAC = tile_n_out*tile_n_in*tile_h_out*fs
            n_4_2 = ((tile_h_out // 8 + ((tile_h_out % 8) > 0)) // 2)
            n_4_1 = ((tile_h_out // 8 + ((tile_h_out % 8) > 0)) % 2)
            cycles_chout_4_2 = (tile_n_out // 4) * (159 + 14 * ((tile_n_in*fs) // 4) + 18 * ((tile_n_in*fs) % 4))
            cycles_leftover_chout_4_2 =  (tile_n_out % 4) * (52 + ((tile_n_in*fs) // 4) * 5 + 8 * ((tile_n_in*fs) % 4))
            cycles_chout_4_1 = (tile_n_out // 4) * (82 + ((tile_n_in*fs) //4) * 9 + 12 * ((tile_n_in*fs) % 4))
            cycles_leftover_chout_4_1 = (tile_n_out % 4) * (28 + 3 * ((tile_n_in*fs) // 4) + 5 * ((tile_n_in*fs) % 4))
            instr = 62 + n_4_2 * (27 + (135 + cycles_chout_4_2 + cycles_leftover_chout_4_2)) + n_4_1 * (21 + (94  + cycles_chout_4_1 + cycles_leftover_chout_4_1))
            constr1 = ( MAC * 1000 ) // instr
            ## left over h
            MAC = tile_n_out*tile_n_in*leftover_tile_h_out*fs
            n_4_2 = ((leftover_tile_h_out // 8 + ((leftover_tile_h_out % 8) > 0)) // 2)
            n_4_1 = ((leftover_tile_h_out // 8 + ((leftover_tile_h_out % 8) > 0)) % 2)
            cycles_chout_4_2 = (tile_n_out // 4) * (159 + 14 * ((tile_n_in*fs) // 4) + 18 * ((tile_n_in*fs) % 4))
            cycles_leftover_chout_4_2 =  (tile_n_out % 4) * (52 + ((tile_n_in*fs) // 4) * 5 + 8 * ((tile_n_in*fs) % 4))
            cycles_chout_4_1 = (tile_n_out // 4) * (82 + ((tile_n_in*fs) //4) * 9 + 12 * ((tile_n_in*fs) % 4))
            cycles_leftover_chout_4_1 = (tile_n_out % 4) * (28 + 3 * ((tile_n_in*fs) // 4) + 5 * ((tile_n_in*fs) % 4))
            instr = 62 + n_4_2 * (27 + (135 + cycles_chout_4_2 + cycles_leftover_chout_4_2)) + n_4_1 * (21 + (94  + cycles_chout_4_1 + cycles_leftover_chout_4_1))
            constr2 = ( MAC * 1000 ) // instr
            ## left over n
            MAC = leftover_tile_nout*tile_n_in*tile_h_out*fs
            n_4_2 = ((tile_h_out // 8 + ((tile_h_out % 8) > 0)) // 2)
            n_4_1 = ((tile_h_out // 8 + ((tile_h_out % 8) > 0)) % 2)
            cycles_chout_4_2 = (leftover_tile_nout // 4) * (159 + 14 * ((tile_n_in*fs) // 4) + 18 * ((tile_n_in*fs) % 4))
            cycles_leftover_chout_4_2 =  (leftover_tile_nout % 4) * (52 + ((tile_n_in*fs) // 4) * 5 + 8 * ((tile_n_in*fs) % 4))
            cycles_chout_4_1 = (leftover_tile_nout // 4) * (82 + ((tile_n_in*fs) //4) * 9 + 12 * ((tile_n_in*fs) % 4))
            cycles_leftover_chout_4_1 = (leftover_tile_nout % 4) * (28 + 3 * ((tile_n_in*fs) // 4) + 5 * ((tile_n_in*fs) % 4))
            instr = 62 + n_4_2 * (27 + (135 + cycles_chout_4_2 + cycles_leftover_chout_4_2)) + n_4_1 * (21 + (94  + cycles_chout_4_1 + cycles_leftover_chout_4_1))
            constr3 = ( MAC * 1000 ) // instr
            ## left over all
            MAC = leftover_tile_nout*tile_n_in*leftover_tile_h_out*fs
            n_4_2 = ((leftover_tile_h_out // 8 + ((leftover_tile_h_out % 8) > 0)) // 2)
            n_4_1 = ((leftover_tile_h_out // 8 + ((leftover_tile_h_out % 8) > 0)) % 2)
            cycles_chout_4_2 = (leftover_tile_nout // 4) * (159 + 14 * ((tile_n_in*fs) // 4) + 18 * ((tile_n_in*fs) % 4))
            cycles_leftover_chout_4_2 =  (leftover_tile_nout % 4) * (52 + ((tile_n_in*fs) // 4) * 5 + 8 * ((tile_n_in*fs) % 4))
            cycles_chout_4_1 = (leftover_tile_nout // 4) * (82 + ((tile_n_in*fs) //4) * 9 + 12 * ((tile_n_in*fs) % 4))
            cycles_leftover_chout_4_1 = (leftover_tile_nout % 4) * (28 + 3 * ((tile_n_in*fs) // 4) + 5 * ((tile_n_in*fs) % 4))
            instr = 62 + n_4_2 * (27 + (135 + cycles_chout_4_2 + cycles_leftover_chout_4_2)) + n_4_1 * (21 + (94  + cycles_chout_4_1 + cycles_leftover_chout_4_1))
            constr4 = ( MAC * 1000 ) // instr
            constr = (constr1*(n_tiles_h-(leftover_tile_h_out>0))*(n_tiles_nout-(leftover_tile_nout>0)) + constr2*(n_tiles_nout-(leftover_tile_nout>0))*(leftover_tile_h_out>0) + constr3*(n_tiles_h-(leftover_tile_h_out>0))*(leftover_tile_nout>0) + constr4)//((n_tiles_h-(leftover_tile_h_out>0))*(n_tiles_nout-(leftover_tile_nout>0)) + (n_tiles_nout-(leftover_tile_nout>0))*(leftover_tile_h_out>0) + (n_tiles_h-(leftover_tile_h_out>0))*(leftover_tile_nout>0) + 1*(leftover_tile_nout>0)*(leftover_tile_h_out>0))
            print(f'Conv no dilation MAC/cycle simulated: {(constr/1000)}')
            return (tile_n_in, tile_n_out, tile_h_in, tile_h_out, (constr/1000), memory/8)
        return None


    def get_tiling_conv_1D_indirect(self, fs,
                        p,
                        stride,
                        dilation,
                        in_channels,
                        out_channels,
                        x_shape,
                        y_shape,
                        BN,
                        buffer_size=44000
                        ):
        s = stride
        n_in = in_channels
        n_out = out_channels
        h_in = x_shape 
        # p = 0
        h_out = y_shape 
        max_tile_n_out = n_out 
        max_tile_n_in = n_in 
        min_tile_h_in = fs
        min_tile_h_out = 1  
        # this is to renormalize all costs
        max_obj_value = sys.maxsize
        # constraints
        input_dim = 8 * n_in * h_in
        output_dim = 8 * n_out * h_out
        weight_dim = 8 * n_in * n_out * fs
        bn_dim = self.BitActivation * n_out * 2
        buffer_total = input_dim + output_dim + weight_dim + bn_dim
        if BN == 0:
            buffer_total -= bn_dim
        if buffer_total <= buffer_size * 8:
            obj_expr = 58+(h_out/2)*(119+(n_out/4)*((14*n_in*fs/4)+159))
            print(f'tile in: {n_in} x {h_in}, tile out: {n_out} x {h_out}' )
            return (n_in, n_out, h_in, h_out, obj_expr, buffer_total/8)
        else:
            db = 2
        parameters = pywrapcp.Solver.DefaultSolverParameters()
        solver = pywrapcp.Solver("simple_CP", parameters)
        # ottimizzato channel first per kernel HWC
        tile_n_in = solver.IntVar(1, max_tile_n_in, 'tile_n_out')
        tile_n_out = solver.IntVar(1, max_tile_n_out, 'tile_n_out')
        tile_h_in = solver.IntVar(min_tile_h_in, h_in, 'tile_h_in') #Temporal
        tile_h_out = solver.IntVar(min_tile_h_out, h_out, 'tile_h_out') #Temporal
        zero_variable = solver.IntVar(0, 0, 'zero variable')
        h_out_intvar = solver.IntVar(min_tile_h_out,h_out,'h_out_intvar')
        solver.Add(h_out_intvar == h_out)
        if ((fs - 1)*dilation+1) <= h_in: #Receptive field size > temporal lenght -> H_in = tile_h_in
            #Adding constraints for geometrical concerns. 
            solver.Add(0 == (tile_h_in - ((fs - 1)*dilation+1)) % s)
            solver.Add(tile_h_out * s == (tile_h_in - (fs - 1)*dilation + (s - 1)))        
            # padding added
            solver.Add(solver.Max((h_in - tile_h_in - (tile_h_in - (fs - 1)*dilation - p)), 0) % (tile_h_in - (fs - 1)*dilation + 1) + abs(solver.Min(
                solver.Max((h_in - tile_h_in - (tile_h_in - (fs - 1)*dilation - p)), 0) % (tile_h_in - (fs - 1)*dilation), 1) - 1) * ((fs - 1)*dilation+1) >= ((fs - 1)*dilation+1))
            solver.Add(h_in >= s*(tile_h_out*(h_out_intvar//tile_h_out)-1)-p+dilation*(fs-1)+1)
        else:
            solver.Add(h_in == tile_h_in )
            solver.Add(h_out == tile_h_out )

        solver.Add(tile_n_in == n_in)

        constr_in = db * 8 * tile_n_in * tile_h_in #* tile_w_in
        constr_out = db * 8 * tile_n_out * tile_h_out #* tile_w_out
        constr_weight = db * 8 * tile_n_in * tile_n_out * fs
        constr_bn = self.BitActivation * n_out * 2
        constraint_all = constr_in + constr_out + constr_weight + constr_bn + 20*32 # 20 are the + 4 added between buffers
        if BN == 0:
            constraint_all -= constr_bn
        solver.Add(constraint_all <= buffer_size * 8)
        # objective
        obj_expr = solver.IntVar(0, max_obj_value, "obj_expr")
        n_tiles_h = solver.IntVar(1, h_in, "n_tiles_h")
        leftover_tile_h_out = solver.IntVar(0, h_out, "leftover_tile_h_out")
        n_tiles_nout = solver.IntVar(1, n_out, "n_tiles_nout")
        leftover_tile_nout = solver.IntVar(0, n_out, "leftover_tile_nout")

        solver.Add(n_tiles_h == (h_out + tile_h_out - 1) // tile_h_out)
        solver.Add(leftover_tile_h_out == (h_out + zero_variable) % tile_h_out)
        solver.Add(n_tiles_nout  == (n_out + tile_n_out - 1) // tile_n_out)
        solver.Add(leftover_tile_nout == (n_out + zero_variable) % tile_n_out)
        ## principal kernel
        solver.Add(obj_expr == (64 * 10000 * tile_n_out
                    + constraint_all
                    + 64 * 1000000 * ((tile_h_out - 1) % 16) #Because of 8 cores -> 2 Pixels
                    + 64 * 1000000 * ((tile_n_out - 1) % 4) #Because of 4x2 computation.
                    + 64 * 10000 * (leftover_tile_nout)
                    + 64 * 10000 * (leftover_tile_nout % 4)
                    + 64 * 10000 * (leftover_tile_h_out  % 16)))
        
        objective = solver.Maximize(obj_expr, 1)

        decision_builder = solver.Phase([tile_n_in, tile_n_out, tile_h_in, tile_h_out, n_tiles_h, leftover_tile_h_out, n_tiles_nout, leftover_tile_nout],
                                        solver.CHOOSE_FIRST_UNBOUND,
                                        solver.ASSIGN_MIN_VALUE)
        
        # Create a solution collector.
        collector = solver.LastSolutionCollector()
        # Add the decision variables.
        collector.Add(tile_n_in)
        collector.Add(tile_n_out)
        collector.Add(tile_h_in)
        collector.Add(tile_h_out)
        collector.Add(n_tiles_h)
        collector.Add(leftover_tile_h_out)
        collector.Add(n_tiles_nout)
        collector.Add(leftover_tile_nout)
        collector.Add(obj_expr)
        collector.Add(constraint_all)
        # Add the objective.
        collector.AddObjective(obj_expr)
        solver.Solve(decision_builder, [objective, collector])
        if collector.SolutionCount() > 0: #Calculating the theoretical cycles to compute this layer.
            best_solution = collector.SolutionCount() - 1
            tile_n_in = collector.Value(best_solution, tile_n_in)
            tile_n_out = collector.Value(best_solution, tile_n_out)
            tile_h_in = collector.Value(best_solution, tile_h_in)
            tile_h_out = collector.Value(best_solution, tile_h_out)
            n_tiles_h = collector.Value(best_solution, n_tiles_h)
            leftover_tile_h_out = collector.Value(best_solution, leftover_tile_h_out)
            n_tiles_nout = collector.Value(best_solution, n_tiles_nout)
            leftover_tile_nout = collector.Value(best_solution, leftover_tile_nout)
            obj_expr = collector.Value(best_solution, obj_expr)
            memory = collector.Value(best_solution, constraint_all)
            if tile_h_in >= h_in:
                tile_h_in = h_in
                tile_h_out = int((tile_h_in -(fs - 1)*dilation + (2*p) + (s - 1))/s)
            MAC = tile_n_out*tile_n_in*tile_h_out*fs
            n_4_2 = ((tile_h_out // 8 + ((tile_h_out % 8) > 0)) // 2)
            n_4_1 = ((tile_h_out // 8 + ((tile_h_out % 8) > 0)) % 2)
            cycles_chout_4_2 = (tile_n_out // 4) * (164 + fs * (29+14*(tile_n_in // 4) + 22 * (tile_n_in % 4)))
            cycles_leftover_chout_4_2 =  (tile_n_out%4)*(49+fs*(15+5*(tile_n_in//4)+8*(tile_n_in%4)))
            cycles_chout_4_1 = (tile_n_out//4)*(90+fs*(15+9*(tile_n_in//4)+17*(tile_n_in%4)))
            cycles_leftover_chout_4_1 = (tile_n_out%4)*(43+fs*(8+2*(tile_n_in//4)+7*(tile_n_in%4)))
            instr = (82+(n_4_2*(62+8*fs+(110+cycles_chout_4_2+cycles_leftover_chout_4_2)) + n_4_1*(36+4*fs+(87+cycles_chout_4_1+cycles_leftover_chout_4_1))))
            constr1 = (MAC * 1000) // instr
            ## left over h
            MAC = leftover_tile_h_out*tile_n_in*tile_n_out*fs
            n_4_2 = ((leftover_tile_h_out // 8 + ((leftover_tile_h_out % 8) > 0)) // 2)
            n_4_1 = ((leftover_tile_h_out // 8 + ((leftover_tile_h_out % 8) > 0)) % 2)
            cycles_chout_4_2 = (tile_n_out // 4) * (164 + fs * (29+14*(tile_n_in // 4) + 22 * (tile_n_in % 4)))
            cycles_leftover_chout_4_2 =  (tile_n_out%4)*(49+fs*(15+5*(tile_n_in//4)+8*(tile_n_in%4)))
            cycles_chout_4_1 = (tile_n_out//4)*(90+fs*(15+9*(tile_n_in//4)+17*(tile_n_in%4)))
            cycles_leftover_chout_4_1 = (tile_n_out%4)*(43+fs*(8+2*(tile_n_in//4)+7*(tile_n_in%4)))
            instr = (82+(n_4_2*(62+8*fs+(110+cycles_chout_4_2+cycles_leftover_chout_4_2)) + n_4_1*(36+4*fs+(87+cycles_chout_4_1+cycles_leftover_chout_4_1))))
            constr2 = (MAC * 1000) // instr
            ## left over n
            MAC = leftover_tile_nout*tile_n_in*tile_h_out*fs
            n_4_2 = ((tile_h_out // 8 + ((tile_h_out % 8) > 0)) // 2)
            n_4_1 = ((tile_h_out // 8 + ((tile_h_out % 8) > 0)) % 2)
            cycles_chout_4_2 = (leftover_tile_nout // 4) * (164 + fs * (29+14*(tile_n_in // 4) + 22 * (tile_n_in % 4)))
            cycles_leftover_chout_4_2 =  (leftover_tile_nout%4)*(49+fs*(15+5*(tile_n_in//4)+8*(tile_n_in%4)))
            cycles_chout_4_1 = (leftover_tile_nout//4)*(90+fs*(15+9*(tile_n_in//4)+17*(tile_n_in%4)))
            cycles_leftover_chout_4_1 = (leftover_tile_nout%4)*(43+fs*(8+2*(tile_n_in//4)+7*(tile_n_in%4)))
            instr = (82+(n_4_2*(62+8*fs+(110+cycles_chout_4_2+cycles_leftover_chout_4_2)) + n_4_1*(36+4*fs+(87+cycles_chout_4_1+cycles_leftover_chout_4_1))))
            constr3 = (MAC * 1000) // instr
            ## left over all
            MAC = leftover_tile_nout*tile_n_in*leftover_tile_h_out*fs
            n_4_2 = ((leftover_tile_h_out // 8 + ((leftover_tile_h_out % 8) > 0)) // 2)
            n_4_1 = ((leftover_tile_h_out // 8 + ((leftover_tile_h_out % 8) > 0)) % 2)
            cycles_chout_4_2 = (leftover_tile_nout // 4) * (164 + fs * (29+14*(tile_n_in // 4) + 22 * (tile_n_in % 4)))
            cycles_leftover_chout_4_2 =  (leftover_tile_nout%4)*(49+fs*(15+5*(tile_n_in//4)+8*(tile_n_in%4)))
            cycles_chout_4_1 = (leftover_tile_nout//4)*(90+fs*(15+9*(tile_n_in//4)+17*(tile_n_in%4)))
            cycles_leftover_chout_4_1 = (leftover_tile_nout%4)*(43+fs*(8+2*(tile_n_in//4)+7*(tile_n_in%4)))
            instr = (82+(n_4_2*(62+8*fs+(110+cycles_chout_4_2+cycles_leftover_chout_4_2)) + n_4_1*(36+4*fs+(87+cycles_chout_4_1+cycles_leftover_chout_4_1))))
            constr4 = (MAC * 1000) // instr
            constr = (constr1*(n_tiles_h-(leftover_tile_h_out>0))*(n_tiles_nout-(leftover_tile_nout>0)) + constr2*(n_tiles_nout-(leftover_tile_nout>0))*(leftover_tile_h_out>0) + constr3*(n_tiles_h-(leftover_tile_h_out>0))*(leftover_tile_nout>0) + constr4)//((n_tiles_h-(leftover_tile_h_out>0))*(n_tiles_nout-(leftover_tile_nout>0)) + (n_tiles_nout-(leftover_tile_nout>0))*(leftover_tile_h_out>0) + (n_tiles_h-(leftover_tile_h_out>0))*(leftover_tile_nout>0) + 1*(leftover_tile_nout>0)*(leftover_tile_h_out>0))
            print(f'Conv indirect MAC/cycle simulated: {(constr/1000)}')
            return (tile_n_in, tile_n_out, tile_h_in, tile_h_out, (constr/1000), memory/8)
        return None


    def get_tiling_conv1d(self, X, Y, W,
                          relu,
                          BN,
                          dilation,
                          has_bias,
                          out_mul, out_shift,
                          type_data='char',
                          full_computation=False,
                          multiple_buffering_factor=2,
                          name='conv',
                          forcing ='None'
                          ):
        # This function generate the layer function to be included in the project for the conv2d operations (Convolutions and Fully Connected layers).
        ds_x = self.BitIn
        ds_y = self.BitOut
        ds_W = self.BitW
        fs1 = self.filter_size[1]
        p_left = self.padding[1]
        p_right = self.padding[3]
        n_in = self.x_shape[0]
        n_out = self.out_ch
        name_include = []
        # L3 tiling
        h_in = self.x_shape[-2]
        w_in = self.x_shape[-1]
        h_out = 1
        if dilation > 1:
            w_out = int(np.floor((w_in - ((fs1 - 1)*dilation) + p_left + p_right + (self.stride - 1)) / self.stride))
        else:
            w_out = int(np.floor((w_in - (fs1 - 1) + p_left + p_right + (self.stride - 1)) / self.stride))
        if p_left==0 and p_right==0 and dilation ==1:
            tiling_MAC_cycle_nodilation = self.get_tiling_conv_1D_nodilation(
                                        fs1,
                                        0,
                                        self.stride,
                                        dilation,
                                        n_in,
                                        n_out,
                                        w_in,
                                        w_out,
                                        BN,
                                        buffer_size=self.buffer_size
                                        )
        tiling_MAC_cycle_normal = self.get_tiling_conv_1D_normal(fs1,
                                    self.padding[1],
                                    self.stride,
                                    dilation,
                                    n_in,
                                    n_out,
                                    w_in,
                                    w_out,
                                    BN,
                                    buffer_size=self.buffer_size
                                    )
        tiling_MAC_cycle_indirect = self.get_tiling_conv_1D_indirect(fs1,
                                    self.padding[1],
                                    self.stride,
                                    dilation,
                                    n_in,
                                    n_out,
                                    w_in,
                                    w_out,
                                    BN,
                                    buffer_size=self.buffer_size
                                    )
        if p_left==0 and p_right==0 and dilation ==1:
            _, _, _, _, MAC_cycle_nodilation, _ = tiling_MAC_cycle_nodilation
        else:
            MAC_cycle_nodilation = 0
        _, _, _, _, MAC_cycle_normal, _ = tiling_MAC_cycle_normal
        _, _, _, _, MAC_cycle_indirect, _ = tiling_MAC_cycle_indirect
        max_MAC = MAC_cycle_nodilation
        layer_type = 'nodilation'
        if p_left==0 and p_right==0 and dilation ==1:
            tiling = tiling_MAC_cycle_nodilation
        if MAC_cycle_normal > max_MAC:
            max_MAC = MAC_cycle_normal
            layer_type = 'normal'
            tiling = tiling_MAC_cycle_normal
        if MAC_cycle_indirect > max_MAC:
            max_MAC = MAC_cycle_indirect
            layer_type = 'indirect'
            tiling = tiling_MAC_cycle_indirect
        ### FOR TEST
        if forcing == 'normal':
            max_MAC = MAC_cycle_normal
            layer_type = 'normal'
            tiling = tiling_MAC_cycle_normal
        elif forcing == 'indirect':
            max_MAC = MAC_cycle_indirect
            layer_type = 'indirect'
            tiling = tiling_MAC_cycle_indirect
        elif forcing == 'nodilation':
            max_MAC = MAC_cycle_nodilation
            layer_type = 'nodilation'
            tiling = tiling_MAC_cycle_nodilation
        if tiling is not None:       
            tile_n_in, tile_n_out, tile_w_in, tile_w_out, MAC_cycle, memory = tiling
            
            x_tot_str = '[%dx%d]' % (n_in, w_in)
            y_tot_str = '[%dx%d]' % (n_out, w_out)
            W_tot_str = '[%dx%dx%d]' % (n_out, n_in, fs1)
            x_tot_size_str = "%.2f KiB" % (1. / 1024. / 8. * (ds_x * n_in * w_in )) if ds_x * \
                n_in * h_in > 1024 else '%d B' % (ds_x * n_in * w_in * 1 / 8.)
            y_tot_size_str = '%.2f KiB' % (1. / 1024. / 8. * (ds_y * n_out * w_out )) if ds_y * \
                 n_out * h_out * w_out > 1024 else '%d B' % (ds_y * n_out * w_out  * 1 / 8.)
            W_tot_size_str = '%.2f KiB' % (1. / 1024. / 8. * (ds_W * n_out * n_in * fs1)) if ds_W * \
                n_out * n_in * fs1 > 1024 else '%d B' % (ds_W * n_out * n_in * fs1 * 1 / 8.)
            x_tile_str = '[%dx%d]' % (tile_n_in, tile_w_in)
            y_tile_str = '[%dx%d]' % (tile_n_out, tile_w_out)
            W_tile_str = '[%dx%dx%d]' % (tile_n_out, tile_n_in, fs1)
            
            x_size_str = "%.2f KiB" % (1. / 1024. / 8. * (ds_x * tile_n_in * tile_w_in )) if ds_x * tile_n_in * tile_w_in > 1024 else '%d B' % (ds_x * tile_n_in * tile_w_in * 1 / 8.)
            y_size_str = '%.2f KiB' % (1. / 1024. / 8. * (ds_y * tile_n_out * tile_w_out)) if ds_y * tile_n_out * tile_w_out > 1024 else '%d B' % (ds_y * tile_n_out * tile_w_out * 1 / 8.)
            y_no_str = '%d' % (max(math.ceil((n_out) / (tile_n_out)), 1) * max(math.ceil((w_out) / (tile_w_out)), 1))
            W_size_str = '%.2f KiB' % (1. / 1024. / 8. * (ds_W * tile_n_out * tile_n_in * fs1)) if (ds_W * tile_n_out * tile_n_in * fs1) > 1024 else '%d B' % (ds_W * tile_n_out * tile_n_in * fs1 * 1 / 8.)
            W_no_str = '%d' % (max(math.ceil((n_out - tile_n_out) / (tile_n_out) + 1), 1) * 1)
            x_no_str = '%d' % (int(int(y_no_str)/int(W_no_str)) * pow(max(math.ceil((n_in - tile_n_in) / (tile_n_in) + 1), 1),2))
            L1_tiles_size = ds_x * tile_n_in * tile_w_in / 8. * (1 + int(int(x_no_str) > 1)) + ds_y * tile_n_out * tile_w_out / 8. * (1 + int(int(y_no_str) > 1)) + n_out * 8 * 2
            L1_tiles_size += (ds_W * tile_n_out * tile_n_in * fs1 / 8.) * (1 + int(int(W_no_str) > 1))
            logging.debug("    L2 size:".ljust(18) + "x: " + x_tot_str.ljust(15) +"y: " + y_tot_str.ljust(15) + "W: " + W_tot_str.ljust(15))
            logging.debug("    L2 buff:".ljust(18) + "x: " + x_tot_size_str.ljust(15) +"y: " + y_tot_size_str.ljust(15) + "W: " + W_tot_size_str.ljust(15))
            logging.debug("    tiles L2-L1:".ljust(18) + "x: " + x_tile_str.ljust(15) +"y: " + y_tile_str.ljust(15) + "W: " + W_tile_str.ljust(15))
            logging.debug("    L1 buff:".ljust(18) + "x: " + x_size_str.ljust(15) +"y: " + y_size_str.ljust(15) + "W: " + W_size_str.ljust(15))
            logging.debug("    no. tiles:".ljust(18) + "x: " + x_no_str.ljust(15) +"y: " + y_no_str.ljust(15) + "W: " + W_no_str.ljust(15))
            logging.debug("    Total L1 occupation:".ljust(18) + str(memory * 1.).ljust(15))
            
            print_template_layer_1D(X, Y, W,
                    n_in, w_in,
                    n_out, w_out,
                    tile_n_in, tile_w_in, tile_w_out,
                    tile_n_out,
                    ds_x, ds_y, ds_W, self.BitActivation, type_data,
                    fs1, p_left, p_right, self.stride,
                    dilation,
                    relu, BN,
                    out_mul, out_shift,
                    name_layer=name,
                    test=False,
                    test_location=self.test_location,
                    has_bias=has_bias,
                    conv_order='PULP-NN',
                    optional='conv',
                    l1_buffer=self.buffer_size,
                    platform=self.platform,
                    chip=self.chip,
                    optional_type=self.optional_type,
                    backend = self.backend,
                    layer_type = layer_type)
            ### L2 memory calculation
            n_out_temp = self.out_ch
            w_in_temp = self.x_shape[-1]
            #h_out_temp = int(np.floor((h_in_temp - (fs1 - 1) + p_left + p_right + (self.stride - 1)) / self.stride))
            w_out_temp = int(np.floor((w_in_temp - ((fs1 - 1)*dilation) + p_left + p_right + (self.stride - 1)) / self.stride))
            out_dim1 = n_out_temp * w_out_temp

            n_in_temp = self.x_shape[0]
            w_in_temp = self.x_shape[-1]
            in_dim1 = n_in_temp * w_in_temp

            n_in_temp = self.x_shape[0]
            n_out_temp = self.out_ch
            weights_dim = n_in_temp * n_out_temp * fs1
            if BN == 1:
                weights_dim +=n_out_temp * int(self.BitActivation / 4)
            return in_dim1, out_dim1, weights_dim, L1_tiles_size, 0, 1, 1, 1
        return None

##############################################
############ L3 TILING #######################
##############################################

    def get_tiling_pool2d_L3(self,
                      BN,
                      input_L3,
                      input_dim_constraint,
                      output_weights_dim_constraint
                      ):
        # tiling for L3-L2 management
        # parameters instantiation
        s = self.stride
        p_top = self.padding[0]
        p_left = self.padding[1]
        p_bottom = self.padding[2]
        p_right = self.padding[3]
        fs1 = self.filter_size[0]
        fs2 = self.filter_size[1]
        conv_overlap_h = 2 * (fs1 // 2) + fs1 % 2 - 1 - (s - 1)
        n_in = self.x_shape[0]
        n_out = self.out_ch
        h_in = self.x_shape[-2] + p_top + p_bottom
        w_in = self.x_shape[-1] + p_left + p_right
        h_out = int(np.floor((h_in - (fs1 - 1) + (s - 1)) / s))
        w_out = int(np.floor((w_in - (fs2 - 1) + (s - 1)) / s))
        h_in = self.x_shape[-2]
        w_in = self.x_shape[-1]
        max_tile_n_out = n_out
        max_tile_n_in = n_in
        min_tile_h_in = fs1
        min_tile_h_out = 1
        # this is to renormalize all costs
        max_obj_value = self.L2_buffer_size * 8 * 32 * 100000
        # constraints
        input_dim = self.BitIn * n_in * h_in * w_in
        output_dim = self.BitOut * n_out * h_out * w_out
        bn_dim = self.BitActivation * n_out * 2
        buffer_total = input_dim + output_dim + bn_dim
        if BN == 0:
            buffer_total -= bn_dim
        ## execute in L2 if constraints are respected
        if (buffer_total <= self.L2_buffer_size * 8) and input_L3==0:
            return (n_in, n_out, h_in, h_out, w_in, w_out)
        else:
            db_O = 1
        # 4 iterations, adding each time a different part to be tiled, either weights, outputs, or both. Input is forced
        for iteration in range(0, 2):
            parameters = pywrapcp.Solver.DefaultSolverParameters()
            solver = pywrapcp.Solver("simple_CP", parameters)
            tile_n_out = solver.IntVar(max_tile_n_out, max_tile_n_out, 'tile_n_out')
            tile_h_out = solver.IntVar(min_tile_h_out, h_out, 'tile_h_out')
            if input_L3 == 0:
                tile_h_in = solver.IntVar(h_in, h_in, 'tile_h_in')
                db_x = 1
            else:
                tile_h_in = solver.IntVar(min_tile_h_in, h_in, 'tile_h_in')
                solver.Add(0 == (tile_h_in - fs1) % s)
                db_x = 2
            if iteration == 0:
                if db_x == 1:
                    db_O = 2
                else:
                    solver.Add(tile_h_out == h_out)
                    db_O = 1
            elif iteration == 1:
                if db_x == 2:
                    db_O = 2
            # L2 constraints on input and output dimension
            if input_dim_constraint > 0:
                solver.Add(db_x * n_in * tile_h_in * w_in <= input_dim_constraint)
            if output_weights_dim_constraint > 0:
                constr_out = db_O * n_out * tile_h_out * w_out
                constraint_all = constr_out
                solver.Add(constraint_all <= output_weights_dim_constraint)
            # scaling is used to ensure datasize is integer
            ds_x_scale = int(math.floor(32 * self.BitIn))
            ds_y_scale = int(math.floor(32 * self.BitOut))
            ds_W_scale = int(math.floor(32 * self.BitW))
            ds_bn_scale = int(math.floor(32 * self.BitActivation))
            # geometrical constraint
            if db_x == 2 and db_O == 2:   
                solver.Add(tile_h_out * s == (tile_h_in - (fs1 - 1) + (s - 1)))
            solver.Add(solver.Max((h_in - tile_h_in - (tile_h_in - fs1 + 1 - p_top)), 0) % (tile_h_in - fs1 + 1) + abs(solver.Min(solver.Max((h_in - tile_h_in - (tile_h_in - fs1 + 1 - p_top)), 0) % (tile_h_in - fs1 + 1), 1) - 1) * fs1 >= fs1)
            constr_in = db_x * ds_x_scale * n_in * tile_h_in * w_in
            constr_out = db_O * ds_y_scale * n_out * tile_h_out * w_out
            constr_bn = ds_bn_scale * n_out * 2
            constraint_all = constr_in + constr_out + constr_bn
            # size constraint
            if BN == 0:
                constraint_all -= constr_bn
            solver.Add(constraint_all <= 32 * self.L2_buffer_size * 8)
            # objective              
            obj_expr = solver.IntVar(0, max_obj_value, "obj_expr")
            # objective function: 
            # 1. constraints for pulp-nn perfromance optimization
            # 2. constraints to have all tiles of same dimension
            solver.Add(obj_expr == constraint_all
                            + 32 * 2 * 100000 * ((tile_h_out - 1) % 8)
                            + 32 * 2 * 1000000 * (((h_in - tile_h_in + p_top) % (tile_h_in - conv_overlap_h )) == 0)
                            + 32 * 2 * 100000 * ((tile_h_in - 1) % 4))
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
                return (n_in, tile_n_out, tile_h_in, tile_h_out, w_in, w_out)
        print("  Pool2D ERROR: no L3-L2 tiling found. Exiting...")
        os._exit(0)
        return None

    def get_tiling_conv2d_L3(self,
                      DW,
                      BN,
                      input_L3,
                      input_dim_constraint,
                      output_weights_dim_constraint,
                      weight_constraint, name
                      ):
        # tiling for L3-L2 management
        # parameters instantiation
        fs1 = self.filter_size[0]
        fs2 = self.filter_size[1]
        s = self.stride
        p_top = self.padding[0]
        p_left = self.padding[1]
        p_bottom = self.padding[2]
        p_right = self.padding[3]
        conv_overlap_h = 2 * (fs1 // 2) + fs1 % 2 - 1 - (s - 1)
        n_in = self.x_shape[0]
        n_out = self.out_ch
        if DW == 1:
            g = self.groups
            n_in = 1
        else:
            g = 1
        h_in = self.x_shape[-2] + p_top + p_bottom
        w_in = self.x_shape[-1] + p_left + p_right
        h_out = int(np.floor((h_in - (fs1 - 1) + (s - 1)) / s))
        w_out = int(np.floor((w_in - (fs2 - 1) + (s - 1)) / s))
        h_in = self.x_shape[-2]
        w_in = self.x_shape[-1]
        max_tile_n_out = n_out
        max_tile_n_in = n_in*g
        min_tile_h_in = fs1
        min_tile_h_out = 1
        # this is to renormalize all costs
        max_obj_value = self.L2_buffer_size * 8 * 32 * 100000
        # constraints
        input_dim = self.BitIn * n_in * g * h_in * w_in
        output_dim = self.BitOut * n_out * h_out * w_out
        weight_dim = self.BitW * n_in * n_out * fs1 * fs2
        bn_dim = self.BitActivation * n_out * 2
        buffer_total = input_dim + output_dim + weight_dim + bn_dim
        if BN == 0:
            buffer_total -= bn_dim
        if weight_constraint > 0:
            if (n_in * n_out * fs1 * fs2 <= weight_constraint):
                flag_weight_ok = True
            else:
                flag_weight_ok = False
        else:
            flag_weight_ok = True
        ## execute in L2 if constraints are respected
        if (buffer_total <= self.L2_buffer_size * 8) and input_L3==0 and flag_weight_ok==True:
            return (n_in, n_out, h_in, h_out, w_in, w_out)
        else:
            db_W = 1
            db_O = 1
        # 4 iterations, adding each time a different part to be tiled, either weights, outputs, or both. Input is forced
        for iteration in range(0, 4):
            parameters = pywrapcp.Solver.DefaultSolverParameters()
            solver = pywrapcp.Solver("simple_CP", parameters)
            tile_n_out = solver.IntVar(1, max_tile_n_out, 'tile_n_out')
            tile_h_out = solver.IntVar(min_tile_h_out, h_out, 'tile_h_out')
            if input_L3 == 0:
                tile_h_in = solver.IntVar(h_in, h_in, 'tile_h_in')
                db_x = 1
            else:
                tile_h_in = solver.IntVar(min_tile_h_in, h_in, 'tile_h_in')
                solver.Add(0 == (tile_h_in - fs1) % s)
                db_x = 2
            if iteration == 0:
                if db_x == 1:
                    db_W = 2
                    db_O = 1
                    solver.Add(tile_h_out == h_out)
                else:
                    solver.Add(tile_h_out == h_out)
                    solver.Add(tile_n_out == n_out)
                    db_W = 1
                    db_O = 1
            elif iteration == 1:
                if db_x == 1:
                    db_W = 1
                    db_O = 2
                    solver.Add(tile_n_out == n_out)
                else:
                    solver.Add(tile_n_out == n_out)
                    db_W = 1
                    db_O = 2
            elif iteration == 2:
                if db_x == 1:
                    db_W = 2
                    db_O = 2
                else:
                    solver.Add(tile_h_out == h_out)
                    db_W = 2
                    db_O = 1
            else:
                db_W = 2
                db_O = 2
            # L2 constraints on input and output dimension
            if input_dim_constraint > 0:
                solver.Add(db_x * n_in * g * tile_h_in * w_in <= input_dim_constraint)
            if output_weights_dim_constraint > 0:
                constr_out = db_O * n_out * tile_h_out * w_out
                if DW == 0:
                    constr_weight = db_W * n_in * tile_n_out * fs1 * fs2
                else:
                    constr_weight = db_W * n_in * g * fs1 * fs2
                if self.BitActivation == 32:
                    constr_bn = tile_n_out * 2 * 4 * db_W
                else:
                    constr_bn = tile_n_out * 2 * 8 * db_W
                constraint_all = constr_out + constr_weight + constr_bn
                solver.Add(constraint_all <= output_weights_dim_constraint)
            if weight_constraint > 0:
                if DW == 0:
                    solver.Add(db_W * n_in * tile_n_out * fs1 * fs2 <= weight_constraint)
                else:
                    solver.Add(db_W * n_in * g * fs1 * fs2 <= weight_constraint)
            # scaling is used to ensure datasize is integer
            ds_x_scale = int(math.floor(32 * self.BitIn))
            ds_y_scale = int(math.floor(32 * self.BitOut))
            ds_W_scale = int(math.floor(32 * self.BitW))
            ds_bn_scale = int(math.floor(32 * self.BitActivation))
            # geometrical constraint
            if db_x == 2 and db_O == 2:   
                solver.Add(tile_h_out * s == (tile_h_in - (fs1 - 1) + (s - 1)))
            solver.Add(solver.Max((h_in - tile_h_in - (tile_h_in - fs1 + 1 - p_top)), 0) % (tile_h_in - fs1 + 1) + abs(solver.Min(solver.Max((h_in - tile_h_in - (tile_h_in - fs1 + 1 - p_top)), 0) % (tile_h_in - fs1 + 1), 1) - 1) * fs1 >= fs1)
            constr_in = db_x * ds_x_scale * n_in * g * tile_h_in * w_in
            constr_out = db_O * ds_y_scale * n_out * tile_h_out * w_out
            if DW == 0:
                constr_weight = db_W * ds_W_scale * n_in * tile_n_out * fs1 * fs2
                constr_weight_L1 = ds_W_scale * n_in * tile_n_out * fs1 * fs2
                constr_im2col_L1 = 32 * 8 * 2 * 8 * fs1 * fs2 * n_in
            else:
                constr_weight = db_W * ds_W_scale * n_in * g * fs1 * fs2
                constr_weight_L1 = ds_W_scale * n_in * g * fs1 * fs2
                constr_im2col_L1 = 32 * 8 * (8 * fs1 * (tile_h_in) + 3) * int( 8 / min(self.BitIn, self.BitOut, self.BitW))
                constr_weight_full_prec_L1 = 32 * 8 * 8 * fs1 * fs2 * int( 8 / self.BitW)
            constr_bn = ds_bn_scale * tile_n_out * 2 * db_W
            constr_in_L1 = ds_x_scale * n_in * g * tile_h_in * w_in
            constr_out_L1 = ds_y_scale * tile_n_out * tile_h_out * w_out
            constr_bn_L1 = ds_bn_scale * tile_n_out * 2
            if 'MatMul' in name or 'Gemm' in name:
                constr_im2col_L1 = 0
            constraint_all = constr_in + constr_out + constr_weight + constr_bn
            constraint_all_L1 = constr_in_L1 + constr_out_L1 + constr_weight_L1 + constr_bn_L1 + constr_im2col_L1
            # size constraint
            if BN == 0:
                constraint_all -= constr_bn
            solver.Add(constraint_all <= 32 * self.L2_buffer_size * 8)
            # objective              
            obj_expr = solver.IntVar(0, max_obj_value, "obj_expr")
            # objective function: 
            # 1. constraints for pulp-nn perfromance optimization
            # 2. constraints to have all tiles of same dimension
            solver.Add(obj_expr == (constraint_all +  32 * 2 * 10000000 * (constraint_all_L1 < self.buffer_size*8*32)
                                    + 32 * 2 * 100000 * ((tile_h_out - 1) % 8)
                                    + 32 * 2 * 100000 * ((tile_n_out) % 4 == 0)
                                    + 32 * 2 * 1000000 * (((n_out - tile_n_out) % tile_n_out) == 0)
                                    + 32 * 2 * 1000000 * (((h_out - tile_h_out) % tile_h_out) == 0)
                                    + 32 * 2 * 1000000 * (((h_in - tile_h_in + p_top) % (tile_h_in - conv_overlap_h )) == 0)
                                    + 32 * 2 * 100000 * (tile_h_out == h_out)
                                    + 32 * 2 * 10000 * (tile_n_out == n_out)))
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
                return (n_in, tile_n_out, tile_h_in, tile_h_out, w_in, w_out)
        print("  Conv2d ERROR: no L3-L2 tiling found. Exiting...")
        os._exit(0)
        return None

    def get_tiling_conv2d_like(self,
                               DW,
                               filter_size1,
                               filter_size2,
                               stride,
                               padding_top,padding_bottom,padding_left,padding_right,
                               groups,
                               BN,
                               in_channels,
                               out_channels,
                               x_shape,
                               y_shape,
                               buffer_size,
                               full_computation=True,
                               multiple_buffering_factor=2,
                               name='conv'): 
        ###############################################
        ##### PARAMETERS INITIALIZATION ###############
        ###############################################
        fs1 = filter_size1
        fs2 = filter_size2
        s = stride
        g = groups
        n_in = in_channels * g
        n_in_weights = in_channels * g
        n_out = out_channels
        h_in = x_shape[-2] + padding_top + padding_bottom
        w_in = x_shape[-1] + padding_left + padding_right
        h_out = y_shape[-2]
        w_out = y_shape[-1]
        h_in = x_shape[-2]
        w_in = x_shape[-1]
        max_tile_n_out = n_out
        max_tile_n_in = n_in
        min_tile_w_in = fs2
        min_tile_h_in = fs1
        min_tile_w_out = 1
        min_tile_h_out = 1
        ###############################################
        ##### L2 DIMENSIONS DEFINITION: EARLY EXIT ####
        ###############################################
        input_dim = self.BitIn * int(n_in/self.number_of_clusters) * h_in * w_in
        if self.backend == 'Occamy':
            input_dim = self.BitIn * int(n_in/self.number_of_clusters) * (h_in + padding_top + padding_bottom) * (w_in + padding_left + padding_right)
            input_dim += ((padding_top + padding_bottom) * (w_in + padding_left + padding_right) + (padding_left + padding_right) * h_in) * self.BitIn * int(n_in/self.number_of_clusters)
        output_dim = self.BitOut * int(n_out/self.number_of_clusters) * h_out * w_out
        if DW == 0:
            weight_dim = self.BitW * n_in * int(n_out/self.number_of_clusters) * fs1 * fs2
        else:
            weight_dim = self.BitW * int(n_out/self.number_of_clusters) * fs1 * fs2
        if DW == 0:
            im2col_dim = 8 * 2 * 8 * fs1 * fs2 * n_in 
        else:
            im2col_dim = 8 * 8 * (fs1 * (h_in + padding_top + padding_bottom) + fs1) * int( 8 / min(self.BitIn, self.BitOut, self.BitW)) 
            weight_full_prec_dim = 8 * 8 * fs1 * fs2 * int( 8 / min(self.BitIn, self.BitOut, self.BitW))
            if self.BitW==8:
                 weight_full_prec_dim = 0
        if 'MatMul' in name or 'Gemm' in name or self.backend == 'Occamy':
            im2col_dim = 0
        bn_dim = self.BitActivation * int(n_out/self.number_of_clusters) * 2
        buffer_total = input_dim + output_dim + weight_dim + im2col_dim + bn_dim
        if n_in >= self.number_of_clusters and self.backend == 'Occamy':
            buffer_total = input_dim * multiple_buffering_factor + output_dim + weight_dim + im2col_dim + bn_dim
        if DW == 1:
            buffer_total+= weight_full_prec_dim
        if BN == 0:
            buffer_total -= bn_dim   
        # return immediatly if the memory fits the L1   
        if buffer_total <= self.buffer_size * 8:
            if fs2 == h_in and h_out == 1:
                h_in = h_in - padding_bottom
            if fs1 == w_in and w_out == 1:
                w_in = w_in - padding_right
            if n_in >= self.number_of_clusters:
                return (int(n_in/self.number_of_clusters), int(n_out/self.number_of_clusters), h_in, h_out, w_in, w_out)
            else:
                return (n_in, int(n_out/self.number_of_clusters), h_in, h_out, w_in, w_out)
        else:
            db = multiple_buffering_factor
        ###############################################
        ##### TILING OF LAYER USING ORTOOLS ###########
        ###############################################
        max_obj_value = self.buffer_size * 8 * 32 * 10000000
        ###############################################
        ##### INITIALIZATION OF THE TILING VARS #######
        ###############################################
        parameters = pywrapcp.Solver.DefaultSolverParameters()
        solver = pywrapcp.Solver("simple_CP", parameters)
        tile_n_in = solver.IntVar(1, max_tile_n_in, 'tile_n_in')
        tile_n_out = solver.IntVar(1, max_tile_n_out, 'tile_n_out')
        tile_h_in = solver.IntVar(min_tile_h_in, h_in, 'tile_h_in')
        if h_in < min_tile_h_in:
            tile_h_in = solver.IntVar(min_tile_h_in, min_tile_h_in, 'tile_h_in')
        tile_h_out = solver.IntVar(min_tile_h_out, h_out, 'tile_h_out')
        tile_w_in = solver.IntVar(min_tile_w_in, w_in, 'tile_w_in')
        if w_in < min_tile_w_in:
            tile_w_in = solver.IntVar(min_tile_w_in, min_tile_w_in, 'tile_w_in')
        tile_w_out = solver.IntVar(min_tile_w_out, w_out, 'tile_w_out')
        zero_variable = solver.IntVar(0, 0, 'zero_variable')
        # scaling is used to ensure datasize is integer
        ds_x_scale = int(math.floor(32 * self.BitIn))
        ds_y_scale = int(math.floor(32 * self.BitOut))
        ds_W_scale = int(math.floor(32 * self.BitW))
        ds_bn_scale = int(math.floor(32 * self.BitActivation))

        ###############################################
        ##### GEOMETRICAL CONSTRAINTS #################
        ###############################################
        if DW != 1 or (h_in > 32 and w_in > 32):
            solver.Add(0 == (tile_h_in - fs1) % s)
            #solver.Add(0 == (tile_w_in - fs2) % s)
        if DW == 1:
            solver.Add(tile_n_in == tile_n_out)
        if DW == 0:
            solver.Add(tile_h_out * s ==(tile_h_in - (fs1 - 1) + (s - 1)))
            solver.Add(tile_w_out * s ==(tile_w_in - (fs2 - 1) + (s - 1)))
        # constraints of border tile. It can't be smaller than filter size
        # solver.Add(solver.Max((h_in - tile_h_in - (tile_h_in - fs1 + 1 - padding_top)), 0) % (tile_h_in - fs1 + 1) + abs(solver.Min(solver.Max((h_in - tile_h_in - (tile_h_in - fs1 + 1 - padding_bottom)), 0) % (tile_h_in - fs1 + 1), 1) - 1) * fs1 >= fs1)
        # solver.Add(solver.Max((w_in - tile_w_in - (tile_w_in - fs2 + 1 - padding_left)), 0) % (tile_w_in - fs2 + 1) + abs(solver.Min(solver.Max((w_in - tile_w_in - (tile_w_in - fs2 + 1 - padding_right)), 0) % (tile_w_in - fs2 + 1), 1) - 1) * fs2 >= fs2)
        ###############################################
        ##### CONSTRAINTS FOR BACKEND LIMITS ##########
        ###############################################
        if DW == 1:
            if h_in <= 32 and w_in <= 32:
                solver.Add(tile_h_in == h_in)
                solver.Add(tile_w_in == w_in)
                solver.Add(tile_h_out == h_out)
                solver.Add(tile_w_out == w_out)
            elif h_in > 32 or w_in > 32:
                solver.Add(tile_h_out * s == (tile_h_in - (fs1 - 1) + ((tile_h_in % h_in) == 0) * (padding_top + padding_bottom) + (s - 1)))
                #solver.Add(tile_w_out * s == (tile_w_in - (fs2 - 1) + ((tile_w_in % w_in) == 0) * (padding_left + padding_right) + (s - 1)))
                solver.Add(tile_w_in == w_in)
                solver.Add(tile_w_out == w_out)
        if DW == 0:
            if n_in >=self.number_of_clusters and (n_in % self.number_of_clusters == 0):
                solver.Add(tile_n_in == int(np.ceil(n_in/self.number_of_clusters)))
            else:
                solver.Add(tile_n_in == int(n_in))
            if self.number_of_clusters>1:
                solver.Add(tile_n_out <= int(n_out/self.number_of_clusters))
        if DW == 1:
            if n_in >=self.number_of_clusters:
                solver.Add(tile_n_out <= int(n_out/self.number_of_clusters))
        # constraint for future mixed
        if DW == 1: 
            solver.Add(tile_n_in % (int(8/min(self.BitIn, self.BitOut, self.BitW)))==0)
        solver.Add(tile_n_out % (int(8/min(self.BitIn, self.BitOut, self.BitW)))==0)
        ###############################################
        ##### CONSTRAINTS FOR DIMENSION ###############
        ###############################################
        constr_in = db * ds_x_scale * tile_n_in * tile_h_in * tile_w_in
        if self.backend == 'Occamy':
            constr_in = db * ds_x_scale * tile_n_in * (tile_h_in + padding_top + padding_bottom) * (tile_w_in + padding_left + padding_right)
            constr_in += db * ds_x_scale * ((padding_top + padding_bottom) * (tile_w_in + padding_left + padding_right) + (padding_left + padding_right) * tile_h_in) * tile_n_in
        constr_out = db * ds_y_scale * tile_n_out * tile_h_out * tile_w_out
        if DW == 0:
            if self.backend == 'Occamy':
                constr_weight = db * ds_W_scale * n_in * tile_n_out * fs1 * fs2
            if self.backend == 'MCU':
                constr_weight = db * ds_W_scale * tile_n_in * tile_n_out * fs1 * fs2
            constr_im2col = 32 * 8 * 2 * 8 * fs1 * fs2 * tile_n_in
        else:
            constr_weight = db * ds_W_scale * n_in * fs1 * fs2
            constr_im2col = 32 * 8 * 8 * ( fs1 * (tile_h_in + padding_top + padding_bottom) + fs1) * int( 8 / min(self.BitIn, self.BitOut, self.BitW))
            constr_weight_full_prec = db * 32 * 8 * 8 * fs1 * fs2 * int( 8 / min(self.BitIn, self.BitOut, self.BitW))
            if self.BitW==8:
                constr_weight_full_prec = 0
        if 'MatMul' in name or 'Gemm' in name or self.backend == 'Occamy':
            constr_im2col = 0
        constr_bn = ds_bn_scale * tile_n_out * 2 * db
        constraint_all = constr_in + constr_out + constr_weight + constr_bn + constr_im2col + 20 
        if DW == 1:
            constraint_all += constr_weight_full_prec
        if BN == 0:
            constraint_all -= constr_bn
        solver.Add(constraint_all <= 32 * self.buffer_size * 8)
        ###############################################
        ##### HEURISTICS ADDITION #####################
        ###############################################
        obj_expr = solver.IntVar(0, max_obj_value, "obj_expr")
        heuristics = 0
        if self.backend != 'Occamy':
            if DW == 0:
                ####### Geometrical Shape of Tiles ############
                heuristics +=  64 * 2000000 * ((tile_h_out - 1) % 8) \
                             + 64 * 3000000 * ((tile_w_out - 1) % 2) \
                             + 64 * 1000000 * ((tile_n_out - 1) % 4) \
                             + 64 * 1000000 * (tile_w_out * tile_h_out >= 16)
                # ####### Total Dimension of Tile ###############
                heuristics += constraint_all
                ####### Maximization of Reuse of im2col #######
                heuristics += 64 * 10000 * tile_n_out \
                            + 64 * 10000 * ((n_out-zero_variable) % (tile_n_out+1))
                ####### Geometrical Shape of Border Tiles #####
                heuristics += 64 * 10000 * ((n_out-zero_variable) % (tile_n_out+1)) \
                            + 64 * 10000 * (((n_out-zero_variable) % (tile_n_out+1)) % 4) \
                            + 64 * 20000 * (((h_out-zero_variable) % (tile_h_out+1)) % 8) \
                            + 64 * 30000 * (((w_out-zero_variable) % (tile_w_out+1)) % 2)
            elif DW == 1:
                ####### Geometrical Shape of Tiles ############
                heuristics += 32 * 10000 * ((tile_n_out > 7)) \
                            + 64 * 10000 * ((tile_n_out - 1) % int(8*8/min(self.BitIn, self.BitOut, self.BitW))) \
                            + 32 * 10000 * ((tile_h_out % 4) == 0)
                ####### Total Dimension of Tile ###############
                heuristics += constraint_all
                ####### Maximization of Reuse of im2col #######
                heuristics += 32 * 1000 * tile_w_out \
                            + 32 * 1000 * tile_h_out \
                            + 32 * 100 * (((h_out-zero_variable) % (tile_h_out+1))) \
                            + 32 * 100 * (((w_out-zero_variable) % (tile_w_out+1)))
                ####### Geometrical Shape of Border Tiles #####
                heuristics += 32 * 100 * (((n_out-zero_variable) % (tile_n_out+1)) > 7) \
                            + 32 * 100 * (((h_out-zero_variable) % (tile_h_out+1)) % 4)
        else:
            ####### Geometrical Shape of Tiles ############
            heuristics += 64 * 10000 * (((((h_out-zero_variable - 1) % tile_h_out)) % 8) > 4) 
            heuristics += 64 * 10000 * (((tile_h_out - 1) % 8) > 4) \
                        + 64 * 10000 * ((tile_n_out - 1) % 8) \
                        + 64 * 10000 * tile_n_out 
            # ####### Total Dimension of Tile ###############
            heuristics += constraint_all
            ####### Geometrical Shape of Border Tiles #####
            #             + 64 * 10000 * ((n_out-zero_variable) % (tile_n_out+1)) \
            #             + 64 * 80000 * (((tile_h_out-zero_variable) % (tile_h_out+1)) > 7)

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
            if tile_h_in >= h_in:
                tile_h_in = h_in
                tile_h_out = int((tile_h_in -(fs1 - 1) + (padding_top + padding_bottom) + (s - 1))/s)
            if tile_w_in >= w_in:
                tile_w_in = w_in
                tile_w_out = int((tile_w_in -(fs2 - 1) + (padding_left + padding_right) + (s - 1))/s)
            return (tile_n_in, tile_n_out, tile_h_in, tile_h_out, tile_w_in, tile_w_out)
        print("  Conv2d ERROR: no L2-L1 tiling found. Exiting...")
        os._exit(0)
        return None

    def get_tiling_conv2d(self, X, Y, W,
                          relu,
                          BN,
                          DW,
                          has_bias,
                          out_mul, out_shift,
                          type_data='char',
                          full_computation=False,
                          multiple_buffering_factor=2,
                          name='conv',
                          input_L3 = 0,
                          input_dim_constraint = 0,
                          output_weights_dim_constraint = 0,
                          weight_constraint = 0
                          ):
        # This function generate the layer function to be included in the project for the conv2d operations (Convolutions and Fully Connected layers).
        ds_x = self.BitIn
        ds_y = self.BitOut
        ds_W = self.BitW
        fs1 = self.filter_size[0]
        fs2 = self.filter_size[1]
        s = self.stride
        p_top = self.padding[0]
        p_left = self.padding[1]
        p_bottom = self.padding[2]
        p_right = self.padding[3]
        n_in = self.x_shape[0]
        n_out = self.out_ch
        name_include = []
        # L3 tiling
        tiling = self.get_tiling_conv2d_L3(DW, BN, input_L3, input_dim_constraint, output_weights_dim_constraint, weight_constraint, name)
        if DW == 1:
            g = self.groups
            n_in = 1
        else:
            g = 1
        h_in = self.x_shape[-2]
        w_in = self.x_shape[-1]
        h_out = int(np.floor((h_in - (fs1 - 1) + p_top + p_bottom + (s - 1)) / s))
        w_out = int(np.floor((w_in - (fs2 - 1) + p_left + p_right + (s - 1)) / s))
        if (n_in, n_out, h_in, h_out, w_in, w_out) == tiling:
            L3_tiling = 0
        else:
            L3_tiling = 1
        # number of L3 tiles identification and dimension for L2 tiles.
        n_in, n_out, h_in, h_out, w_in, w_out = tiling
        factor_ch_out = self.out_ch/n_out
        factor_h_out = int(np.ceil(np.floor((self.x_shape[-2] - (fs1 - 1) + p_top + p_bottom + (s - 1)) / s)/h_out))
        conv_overlap_h = 2 * (fs1 // 2) + fs1 % 2 - 1 - (s - 1)
        if (self.x_shape[-2] - h_in)==0:
            factor_h_in = 1
        else:
            factor_h_in = 1 + int(np.ceil((self.x_shape[-2] + p_top + p_bottom - h_in) / (h_in - conv_overlap_h ))) 
            if p_bottom > 0 and (self.x_shape[-2] + p_top - h_in) % (h_in - conv_overlap_h ) == 0 and (h_in - conv_overlap_h ) != 1:
                factor_h_in = 1 + int((self.x_shape[-2] + p_top - h_in) / (h_in - conv_overlap_h ))
        # report
        if L3_tiling == 1:
            h_in_L3 = self.x_shape[-2]
            h_out_L3 = (int(np.floor((self.x_shape[-2] - (fs1 - 1) + p_top + p_bottom + (s - 1)) / s)))
            ch_out_L3 = self.out_ch
            x_tot_str = '[%dx%dx%d]' % (g * n_in, h_in_L3, w_in)
            y_tot_str = '[%dx%dx%d]' % (ch_out_L3, h_out_L3, w_out)
            W_tot_str = '[%dx%dx%dx%d]' % (ch_out_L3, n_in, fs1, fs2)

            x_tot_size_str = "%.2f KiB" % (1. / 1024. / 8. * (ds_x * g * n_in * h_in_L3 * w_in)) if ds_x * \
                g * n_in * h_in_L3 * w_in > 1024 else '%d B' % (ds_x * g * n_in * h_in_L3 * w_in * 1 / 8.)
            y_tot_size_str = '%.2f KiB' % (1. / 1024. / 8. * (ds_y * ch_out_L3 * h_out_L3 * w_out)) if ds_y * \
                g * ch_out_L3 * h_out_L3 * w_out > 1024 else '%d B' % (ds_y * ch_out_L3 * h_out_L3 * w_out * 1 / 8.)
            if g > 1:
                W_tot_size_str = '%.2f KiB' % (1. / 1024. / 8. * (ds_W * ch_out_L3 * fs1 * fs2)) if ds_W * \
                    n_out * n_in * fs1 * fs2 > 1024 else '%d B' % (ds_W * ch_out_L3 * n_in * fs1 * fs2 * 1 / 8.)
            else:
                W_tot_size_str = '%.2f KiB' % (1. / 1024. / 8. * (ds_W * ch_out_L3 * n_in * fs1 * fs2)) if ds_W * \
                    ch_out_L3 * n_in * fs1 * fs2 > 1024 else '%d B' % (ds_W * ch_out_L3 * n_in * fs1 * fs2 * 1 / 8.)
            tile_n_in, tile_n_out, tile_h_in, tile_h_out, tile_w_in, tile_w_out = n_in, n_out, h_in, h_out, w_in, w_out
            x_tile_str = '[%dx%dx%d]' % (g * tile_n_in, tile_h_in, tile_w_in)
            y_tile_str = '[%dx%dx%d]' % (ch_out_L3, tile_h_out, tile_w_out)
            if g > 1:
                W_tile_str = '[%dx1x%dx%d]' % (tile_n_out, fs1, fs2)
            else:
                W_tile_str = '[%dx%dx%dx%d]' % (tile_n_out, tile_n_in, fs1, fs2)
            x_size_str = "%.2f KiB" % (1. / 1024. / 8. * (ds_x * g * tile_n_in * tile_h_in * tile_w_in)) if ds_x * g * \
                tile_n_in * tile_h_in * tile_w_in > 1024 else '%d B' % (ds_x * g * tile_n_in * tile_h_in * tile_w_in * 1 / 8.)
            y_size_str = '%.2f KiB' % (1. / 1024. / 8. * (ds_y * ch_out_L3 * tile_h_out * tile_w_out)) if ds_y * \
                ch_out_L3 * tile_h_out * tile_w_out > 1024 else '%d B' % (ds_y * ch_out_L3 * tile_h_out * tile_w_out * 1 / 8.)
            y_no_str = '%d' % (max(math.ceil((n_out) / (tile_n_out)), 1) * max(math.ceil(
                (h_out_L3) / (tile_h_out)), 1) * max(math.ceil((w_out) / (tile_w_out)), 1))
            if g > 1:
                W_size_str = '%.2f KiB' % (1. / 1024. / 8. * (ds_W * tile_n_out * fs1 * fs2)) if ds_W * \
                    tile_n_out * fs1 * fs2 > 1024 else '%d B' % (ds_W * tile_n_out * fs1 * fs2 * 1 / 8.)
                W_no_str = '%d' % (max(math.ceil((ch_out_L3 - tile_n_out) / (tile_n_out) + 1), 1)
                                   * max(math.ceil((n_in - tile_n_in) / (tile_n_in) + 1), 1))
            else:
                W_size_str = '%.2f KiB' % (1. / 1024. / 8. * (ds_W * tile_n_out * tile_n_in * fs1 * fs2)) if ds_W * \
                    tile_n_out * tile_n_in * fs1 * fs2 > 1024 else '%d B' % (ds_W * tile_n_out * tile_n_in * fs1 * fs2 * 1 / 8.)
                W_no_str = '%d' % (max(math.ceil((ch_out_L3 - tile_n_out) / (tile_n_out) + 1), 1) * 1)
            x_no_str = '%d' % (factor_h_in)
            L2_tiles_size = ds_x * g * tile_n_in * tile_h_in * tile_w_in / 8. * (1 + int(factor_h_in > 1)) + ds_y * ch_out_L3 * tile_h_out * tile_w_out / 8. * (1 + int(factor_h_out > 1))
            if g > 1:
                L2_tiles_size += (ds_W * tile_n_out * fs1 * fs2 / 8. + tile_n_out * 8) * (1 + int(int(W_no_str) > 1))
            else:
                L2_tiles_size += (ds_W * tile_n_out * tile_n_in * fs1 * fs2 / 8. + tile_n_out * 8) * (1 + int(int(W_no_str) > 1))
            if g > 1:
                logging.debug("  Conv2d Depthwise tiling:")
            else:
                logging.debug("  Conv2d Pointwise tiling:")
            logging.debug(f"  Precisions. x = {ds_x} bit, y = {ds_y} bit, W = {ds_W} bit")
            logging.debug("    L3 size:".ljust(18) + "x: " + x_tot_str.ljust(15) +
                          "y: " + y_tot_str.ljust(15) + "W: " + W_tot_str.ljust(15))
            logging.debug("    L3 buff:".ljust(18) + "x: " + x_tot_size_str.ljust(15) +
                          "y: " + y_tot_size_str.ljust(15) + "W: " + W_tot_size_str.ljust(15))
            logging.debug("    tiles L3-L2:".ljust(18) + "x: " + x_tile_str.ljust(15) +
                          "y: " + y_tile_str.ljust(15) + "W: " + W_tile_str.ljust(15))
            logging.debug("    L2 buff:".ljust(18) + "x: " + x_size_str.ljust(15) +
                          "y: " + y_size_str.ljust(15) + "W: " + W_size_str.ljust(15))
            logging.debug("    no. tiles:".ljust(18) + "x: " + x_no_str.ljust(15) +
                          "y: " + y_no_str.ljust(15) + "W: " + W_no_str.ljust(15))
            logging.debug("    Total L2 occupation:".ljust(18) + str(L2_tiles_size).ljust(15))
            if factor_h_in > 1 and factor_h_out > 1:
                logging.debug("    Tiling Input Act. and Output Act.")
            elif factor_h_in > 1:
                logging.debug("    Tiling Input Act.")
            elif factor_h_out > 1:
                logging.debug("    Tiling Output Act.")
            if int(W_no_str) > 1:
                logging.debug("    Tiling Weights")
            if int(W_no_str) > 1 and (factor_h_in > 1 or factor_h_out > 1):
                print("Convolution: Tiling of weights and Input/output activation from L3 not yet working. Exiting...")
                print(f"Input Constraint: {input_dim_constraint}, Output+Weights Constraint: {output_weights_dim_constraint}, Weight constraint: {weight_constraint}")
                os._exit(0)
        else:
            if g > 1:
                logging.debug("  Conv2d Depthwise tiling:")
            else:
                logging.debug("  Conv2d Pointwise tiling:")
            logging.debug("  No L3 tiling")
        # tiling of L2-L1. It is either the only tiling problem, or the first one for different L2 dimensions (top, middle, bottom)
        if factor_h_in > 1:
            h_out = int(np.floor((h_in - (fs1 - 1) + (s - 1)) / s))
        if factor_h_out > 1:
            h_in = h_out * s + (fs1 - 1) - (s - 1)
        if (p_top + p_bottom) > 0 and (factor_h_in > 1 or factor_h_out > 1):
            tiling = self.get_tiling_conv2d_like(
                DW,
                fs1,
                fs2,
                s,
                0, 0, p_left, p_right,
                g,
                BN,
                n_in,
                n_out,
                [n_in, h_in, w_in],
                [n_out, h_out, w_out],
                self.buffer_size,
                full_computation=full_computation,
                multiple_buffering_factor=multiple_buffering_factor,
                name=name)
        else:
            tiling = self.get_tiling_conv2d_like(
                DW,
                fs1,
                fs2,
                s,
                p_top,p_bottom,p_left,p_right,
                g,
                BN,
                n_in,
                n_out,
                [n_in, h_in, w_in],
                [n_out, h_out, w_out],
                self.buffer_size,
                full_computation=full_computation,
                multiple_buffering_factor=multiple_buffering_factor,
                name=name)        
        name_include.append(name)
        # report
        if tiling is not None:

            tile_n_in, tile_n_out, tile_h_in, tile_h_out, tile_w_in, tile_w_out = tiling

            x_tot_str = '[%dx%dx%d]' % (g * n_in, h_in, w_in)
            y_tot_str = '[%dx%dx%d]' % (n_out, h_out, w_out)
            W_tot_str = '[%dx%dx%dx%d]' % (n_out, n_in, fs1, fs2)

            x_tot_size_str = "%.2f KiB" % (1. / 1024. / 8. * (ds_x * g * n_in * h_in * w_in)) if ds_x * \
                g * n_in * h_in * w_in > 1024 else '%d B' % (ds_x * g * n_in * h_in * w_in * 1 / 8.)
            y_tot_size_str = '%.2f KiB' % (1. / 1024. / 8. * (ds_y * n_out * h_out * w_out)) if ds_y * \
                g * n_out * h_out * w_out > 1024 else '%d B' % (ds_y * n_out * h_out * w_out * 1 / 8.)
            if g > 1:
                W_tot_size_str = '%.2f KiB' % (1. / 1024. / 8. * (ds_W * n_out * fs1 * fs2)) if ds_W * \
                    n_out * n_in * fs1 * fs2 > 1024 else '%d B' % (ds_W * n_out * n_in * fs1 * fs2 * 1 / 8.)
            else:
                W_tot_size_str = '%.2f KiB' % (1. / 1024. / 8. * (ds_W * n_out * n_in * fs1 * fs2)) if ds_W * \
                    n_out * n_in * fs1 * fs2 > 1024 else '%d B' % (ds_W * n_out * n_in * fs1 * fs2 * 1 / 8.)

            x_tile_str = '[%dx%dx%d]' % (tile_n_in, tile_h_in, tile_w_in)
            y_tile_str = '[%dx%dx%d]' % (tile_n_out, tile_h_out, tile_w_out)
            if g > 1:
                W_tile_str = '[%dx1x%dx%d]' % (tile_n_out, fs1, fs2)
            else:
                W_tile_str = '[%dx%dx%dx%d]' % (tile_n_out, tile_n_in, fs1, fs2)
            x_size_str = "%.2f KiB" % (1. / 1024. / 8. * (ds_x * tile_n_in * tile_h_in * tile_w_in)) if ds_x * \
                tile_n_in * tile_h_in * tile_w_in > 1024 else '%d B' % (ds_x * tile_n_in * tile_h_in * tile_w_in * 1 / 8.)
            y_size_str = '%.2f KiB' % (1. / 1024. / 8. * (ds_y * tile_n_out * tile_h_out * tile_w_out)) if ds_y * \
                tile_n_out * tile_h_out * tile_w_out > 1024 else '%d B' % (ds_y * tile_n_out * tile_h_out * tile_w_out * 1 / 8.)
            y_no_str = '%d' % (max(math.ceil((n_out) / (tile_n_out)), 1) * max(math.ceil(
                (h_out) / (tile_h_out)), 1) * max(math.ceil((w_out) / (tile_w_out)), 1))
            if g > 1:
                W_size_str = '%.2f KiB' % (1. / 1024. / 8. * (ds_W * tile_n_out * fs1 * fs2)) if ds_W * \
                    tile_n_out * fs1 * fs2 > 1024 else '%d B' % (ds_W * tile_n_out * fs1 * fs2 * 1 / 8.)
                W_no_str = '%d' % (max(math.ceil((n_out - tile_n_out) / (tile_n_out) + 1), 1)
                                   * max(math.ceil((n_in - tile_n_in) / (tile_n_in) + 1), 1))
            else:
                W_size_str = '%.2f KiB' % (1. / 1024. / 8. * (ds_W * tile_n_out * tile_n_in * fs1 * fs2)) if ds_W * \
                    tile_n_out * tile_n_in * fs1 * fs2 > 1024 else '%d B' % (ds_W * tile_n_out * tile_n_in * fs1 * fs2 * 1 / 8.)
                W_no_str = '%d' % (max(math.ceil((n_out - tile_n_out) / (tile_n_out) + 1), 1) * 1)
            x_no_str = '%d' % (int(int(y_no_str)/int(W_no_str)) * pow(max(math.ceil((n_in*g - tile_n_in) / (tile_n_in) + 1), 1),2))
            L1_tiles_size = ds_x * tile_n_in * tile_h_in * tile_w_in / 8. * (1 + int(int(x_no_str) > 1)) + ds_y * tile_n_out * tile_h_out * tile_w_out / 8. * (1 + int(int(y_no_str) > 1)) + tile_n_out * 8 * 2
            if g > 1:
                L1_tiles_size += (ds_W * tile_n_out * fs1 * fs2 / 8.) * (1 + int(int(W_no_str) > 1))
                L1_tiles_size += (8 * fs1 * (tile_h_in) + 3) * int( 8 / min(self.BitIn, self.BitOut, self.BitW))
            else:
                L1_tiles_size += (ds_W * tile_n_out * tile_n_in * fs1 * fs2 / 8.) * (1 + int(int(W_no_str) > 1))
                if(BN == 1):
                    L1_tiles_size += (8 * tile_n_in * fs1 * fs2 / 8. * 8 * 2.)
            if g > 1:
                logging.debug("    groups:".ljust(18) + '%d' % g)
            logging.debug("    L2 size:".ljust(18) + "x: " + x_tot_str.ljust(15) +
                          "y: " + y_tot_str.ljust(15) + "W: " + W_tot_str.ljust(15))
            logging.debug("    L2 buff:".ljust(18) + "x: " + x_tot_size_str.ljust(15) +
                          "y: " + y_tot_size_str.ljust(15) + "W: " + W_tot_size_str.ljust(15))
            logging.debug("    tiles L2-L1:".ljust(18) + "x: " + x_tile_str.ljust(15) +
                          "y: " + y_tile_str.ljust(15) + "W: " + W_tile_str.ljust(15))
            logging.debug("    L1 buff:".ljust(18) + "x: " + x_size_str.ljust(15) +
                          "y: " + y_size_str.ljust(15) + "W: " + W_size_str.ljust(15))
            logging.debug("    no. tiles:".ljust(18) + "x: " + x_no_str.ljust(15) +
                          "y: " + y_no_str.ljust(15) + "W: " + W_no_str.ljust(15))
            logging.debug("    Total L1 occupation:".ljust(18) + str(L1_tiles_size * 1.).ljust(15))
            # printing layer .c file. Either a unique one, or top,bottom and middle one (for which also tiling is computed).
            if (p_top+p_bottom) > 0 and (factor_h_in > 1 or factor_h_out > 1):
                in_dim1, out_dim1, weight_dim1, l2_dim_k, l2_dim_lambda, bias_dim1, l1_dim1, n_out1, w_out1, h_out1 = print_template_layer(
                    X, Y, W,
                    n_in * g, h_in, w_in,
                    n_out, h_out, w_out,
                    tile_n_in, tile_h_in, tile_w_in, tile_h_out, tile_w_out,
                    tile_n_out,
                    ds_x, ds_y, ds_W, self.BitActivation, type_data,
                    fs1, fs2, 0,0, p_left, p_right, s,
                    relu, BN, DW,
                    out_mul, 0, out_shift, factor_ch_out, factor_h_out, factor_h_in,
                    name_layer=name,
                    test=False,
                    test_location=self.test_location,
                    has_bias=has_bias,
                    conv_order='PULP-NN',
                    optional='conv',
                    l1_buffer=self.buffer_size,
                    platform=self.platform,
                    chip=self.chip,
                    optional_type=self.optional_type,
                    L3_tiling = L3_tiling,
                    sdk = self.sdk,
                    backend = self.backend,
                    number_of_clusters = self.number_of_clusters,
                    dma_parallelization = self.dma_parallelization)
            else:
                in_dim1, out_dim1, weight_dim1, l2_dim_k, l2_dim_lambda, bias_dim1, l1_dim1, n_out1, w_out1, h_out1 = print_template_layer(
                    X, Y, W,
                    n_in * g, h_in, w_in,
                    n_out, h_out, w_out,
                    tile_n_in, tile_h_in, tile_w_in, tile_h_out, tile_w_out,
                    tile_n_out,
                    ds_x, ds_y, ds_W, self.BitActivation, type_data,
                    fs1, fs2, p_top,p_bottom, p_left, p_right, s,
                    relu, BN, DW,
                    out_mul, 0, out_shift, factor_ch_out, factor_h_out, factor_h_in,
                    name_layer=name,
                    test=False,
                    test_location=self.test_location,
                    has_bias=has_bias,
                    conv_order='PULP-NN',
                    optional='conv',
                    l1_buffer=self.buffer_size,
                    platform=self.platform,
                    chip=self.chip,
                    optional_type=self.optional_type,
                    L3_tiling = L3_tiling,
                    sdk = self.sdk,
                    backend = self.backend,
                    number_of_clusters = self.number_of_clusters,
                    dma_parallelization = self.dma_parallelization)   
            if (p_top + p_bottom) > 0 and (factor_h_in > 1 or factor_h_out > 1):
                tiling = self.get_tiling_conv2d_like(
                    DW,
                    fs1,
                    fs2,
                    s,
                    p_top, 0, p_left, p_right,
                    g,
                    BN,
                    n_in,
                    n_out,
                    [n_in, h_in, w_in],
                    [n_out, h_out, w_out],
                    self.buffer_size,
                    full_computation=full_computation,
                    multiple_buffering_factor=multiple_buffering_factor,
                    name=name) 
                tile_n_in, tile_n_out, tile_h_in, tile_h_out, tile_w_in, tile_w_out = tiling
                in_dim1, out_dim2, weight_dim1, l2_dim_k, l2_dim_lambda, bias_dim1, l1_dim1, n_out1, w_out1, h_out1 = print_template_layer(
                    X, Y, W,
                    n_in * g, h_in, w_in,
                    n_out, h_out, w_out,
                    tile_n_in, tile_h_in, tile_w_in, tile_h_out, tile_w_out,
                    tile_n_out,
                    ds_x, ds_y, ds_W, self.BitActivation, type_data,
                    fs1, fs2, p_top, 0, p_left, p_right, s,
                    relu, BN, DW,
                    out_mul, 0, out_shift, factor_ch_out, factor_h_out, factor_h_in,
                    name_layer=name + '_p_t',
                    test=False,
                    test_location=self.test_location,
                    has_bias=has_bias,
                    conv_order='PULP-NN',
                    optional='conv',
                    l1_buffer=self.buffer_size,
                    platform=self.platform,
                    chip=self.chip,
                    optional_type=self.optional_type,
                    L3_tiling = L3_tiling,
                    sdk = self.sdk,
                    backend = self.backend,
                    number_of_clusters = self.number_of_clusters,
                    dma_parallelization = self.dma_parallelization) 
                if out_dim2 > out_dim1:
                    out_dim1 = out_dim2     
                h_in_last = h_in
                h_out_last = int(np.floor((h_in_last + p_bottom - (fs1 - 1) + (s - 1)) / s))
                #### CHECK WELL especially second nested if
                if factor_h_in > 2 or factor_h_out > 2:
                    if ((self.x_shape[-2] - h_in - h_in + conv_overlap_h + p_top) % (h_in - conv_overlap_h )) != 0:
                        h_in_last = ((self.x_shape[-2] - h_in - h_in + conv_overlap_h + p_top) % (h_in - conv_overlap_h )) + conv_overlap_h
                        h_out_last = int(np.floor((h_in_last + p_bottom - (fs1 - 1) + (s - 1)) / s))
                    elif (h_in - conv_overlap_h ) == 1:
                        h_in_last = h_in - 1
                        h_out_last = int(np.floor((h_in_last + p_bottom - (fs1 - 1) + (s - 1)) / s))
                    pad_bot = p_bottom - ((self.x_shape[-2] - h_in - h_in + conv_overlap_h + p_top + p_bottom) % (h_in - conv_overlap_h ))
                elif factor_h_in > 1 or factor_h_out > 1:
                    if ((self.x_shape[-2] - h_in) % (h_in - conv_overlap_h -p_top)) != 0:
                        h_in_last = ((self.x_shape[-2] - h_in) % (h_in - conv_overlap_h -p_top)) + conv_overlap_h + p_bottom
                        h_out_last = int(np.floor((h_in_last + p_bottom - (fs1 - 1) + (s - 1)) / s))
                    elif (h_in - conv_overlap_h ) == 1:
                        h_in_last = h_in - 1
                        h_out_last = int(np.floor((h_in_last + p_bottom - (fs1 - 1) + (s - 1)) / s))
                    pad_bot = p_bottom - ((self.x_shape[-2] - h_in) % (h_in - conv_overlap_h -p_top))
                tiling = self.get_tiling_conv2d_like(
                    DW,
                    fs1,
                    fs2,
                    s,
                    0, pad_bot, p_left, p_right,
                    g,
                    BN,
                    n_in,
                    n_out,
                    [n_in, h_in_last, w_in],
                    [n_out, h_out_last, w_out],
                    self.buffer_size,
                    full_computation=full_computation,
                    multiple_buffering_factor=multiple_buffering_factor,
                    name=name)  
                tile_n_in, tile_n_out, tile_h_in, tile_h_out, tile_w_in, tile_w_out = tiling
                in_dim1, out_dim2, weight_dim1, l2_dim_k, l2_dim_lambda, bias_dim1, l1_dim1, n_out1, w_out1, h_out1 = print_template_layer(
                    X, Y, W,
                    n_in * g, h_in_last, w_in,
                    n_out, h_out_last, w_out,
                    tile_n_in, tile_h_in, tile_w_in, tile_h_out, tile_w_out,
                    tile_n_out,
                    ds_x, ds_y, ds_W, self.BitActivation, type_data,
                    fs1, fs2,0, p_bottom, p_left, p_right, s,
                    relu, BN, DW,
                    out_mul, 0, out_shift, factor_ch_out, factor_h_out, factor_h_in,
                    name_layer=name + '_p_b',
                    test=False,
                    test_location=self.test_location,
                    has_bias=has_bias,
                    conv_order='PULP-NN',
                    optional='conv',
                    l1_buffer=self.buffer_size,
                    platform=self.platform,
                    chip=self.chip,
                    optional_type=self.optional_type,
                    L3_tiling = L3_tiling,
                    sdk = self.sdk,
                    backend = self.backend,
                    number_of_clusters = self.number_of_clusters,
                    dma_parallelization = self.dma_parallelization)
                if out_dim2 > out_dim1:
                    out_dim1 = out_dim2   
                name_include.append(name + '_p_t')
                name_include.append(name + '_p_b')                   
            if self.test_location == 'L3_partial':
                full_net = 0
            else:
                full_net = 1 
            # print template layer for L3 execution of the layer, if present.
            if L3_tiling == 1 or input_L3 == 1:
                print_template_layer_L3(
                    X, W, Y, fs1, fs2, p_top, s,
                    self.BitIn, self.BitW, self.BitOut,
                    factor_ch_out, 
                    factor_h_out, 
                    factor_h_in,
                    name_include,
                    int(n_out * w_out * h_out * self.BitOut / 8),
                    int(n_in * g * w_in * h_in * self.BitIn / 8),
                    n_in * g * self.x_shape[-2] * self.x_shape[-1],
                    weight_dim1,
                    l2_dim_lambda,
                    l2_dim_k,
                    w_out,
                    h_out,
                    n_out,
                    w_in,
                    h_in,
                    n_in * g,
                    full_net,
                    self.platform,
                    ds_x,
                    ds_y,
                    self.test_location,
                    out_mul, out_shift,
                    self.buffer_size,
                    input_L3,
                    backend = self.backend)
            ### L2 memory calculation
            if factor_h_out > 1:
                out_dim1 = out_dim1*2
            else:
                n_out_temp = self.out_ch
                h_in_temp = self.x_shape[-2]
                w_in_temp = self.x_shape[-1]
                h_out_temp = int(np.floor((h_in_temp - (fs1 - 1) + p_top + p_bottom + (s - 1)) / s))
                w_out_temp = int(np.floor((w_in_temp - (fs2 - 1) + p_left + p_right + (s - 1)) / s))
                out_dim1 = n_out_temp * h_out_temp * w_out_temp
            if factor_h_in > 1:
                in_dim1 = in_dim1*2
            else:
                n_in_temp = self.x_shape[0]
                h_in_temp = self.x_shape[-2]
                w_in_temp = self.x_shape[-1]
                in_dim1 = n_in_temp * h_in_temp * w_in_temp
            if factor_ch_out > 1:
                weights_dim = ( weight_dim1 + l2_dim_lambda + l2_dim_k + bias_dim1 ) * 2
            else:
                n_in_temp = self.x_shape[0]
                n_out_temp = self.out_ch
                if self.groups > 1:
                    weights_dim = int(n_in_temp * fs1 *fs2 * self.BitW / 8) + bias_dim1
                else:
                    weights_dim = int(n_in_temp * n_out_temp * fs1 *fs2 * self.BitW / 8) + bias_dim1
                if BN == 1:
                    weights_dim +=n_out_temp * int(self.BitActivation / 4)
            return in_dim1, out_dim1, weights_dim, l1_dim1, L3_tiling, factor_ch_out, factor_h_out, factor_h_in
        return None

    def get_tiling_pool2d(self, X, Y, W,
                          type_data='char',
                          relu=0,  BN = 0,
                          out_mul=0,
                          out_shift=0,
                          name='pool2d',
                          input_L3 = 0,
                          input_dim_constraint = 0,
                          output_weights_dim_constraint = 0,
                          type='Avg'
                          ): 
        # This function generate the layer function to be included in the project for the pooling operation.
        parameters = pywrapcp.Solver.DefaultSolverParameters()
        name_include = []
        solver = pywrapcp.Solver("simple_CP", parameters)
        cost_w = 10
        cost_h = 1
        cost_n = 1000
        cost_dim = 10000
        ds_x = self.BitIn
        ds_y = self.BitOut
        fs1 = self.filter_size[0]
        fs2 = self.filter_size[1]
        s = self.stride
        p_top = self.padding[0]
        p_left = self.padding[1]
        p_bottom = self.padding[2]
        p_right = self.padding[3]
        n_in = self.x_shape[0]
        h_in = self.x_shape[-2] + p_top + p_bottom
        w_in = self.x_shape[-1] + p_left + p_right
        h_out = int(np.floor((h_in - (fs1 - 1) + (s - 1)) / s))
        w_out = int(np.floor((w_in - (fs2 - 1) + (s - 1)) / s))
        h_in = self.x_shape[-2]
        w_in = self.x_shape[-1]
        n_out = n_in
        max_tile_n_out = n_out
        max_tile_n_in = n_in
        min_tile_w_in = fs2
        min_tile_h_in = fs1
        min_tile_w_out = 1
        min_tile_h_out = 1
        # this is to renormalize all costs
        max_obj_value = self.buffer_size * 10000 * 8 * 32
        memory = self.BitIn * n_in * h_in * w_in + self.BitOut * n_out * h_out * w_out
        if memory >= self.L2_buffer_size * 8:
            tiling = self.get_tiling_pool2d_L3(BN, input_L3, input_dim_constraint, output_weights_dim_constraint)
            # number of L3 tiles identification and dimension for L2 tiles.
            n_in, n_out, h_in, h_out, w_in, w_out = tiling
        if memory >= self.L2_buffer_size * 8:
            L3_tiling = 1
        else:
            L3_tiling = 0
        factor_ch_out = self.out_ch/n_out
        factor_h_out = (int(np.floor((self.x_shape[-2] - (fs1 - 1) + p_top + p_bottom + (s - 1)) / s)))/h_out
        conv_overlap_h = 2 * (fs1 // 2) + fs1 % 2 - 1 - (s - 1)
        if (self.x_shape[-2] - h_in)==0:
            factor_h_in = 1
        else:
            factor_h_in = 1 + int(np.floor((self.x_shape[-2] - h_in + p_top + p_bottom) / (h_in - conv_overlap_h ))) 
        # report
        if L3_tiling == 1:
            h_in_L3 = self.x_shape[-2]
            h_out_L3 = (int(np.floor((self.x_shape[-2] - (fs1 - 1) + p_top + p_bottom + (s - 1)) / s)))
            ch_out_L3 = self.out_ch
            x_tot_str = '[%dx%dx%d]' % (n_in, h_in_L3, w_in)
            y_tot_str = '[%dx%dx%d]' % (ch_out_L3, h_out_L3, w_out)

            x_tot_size_str = "%.2f KiB" % (1. / 1024. / 8. * (ds_x * n_in * h_in_L3 * w_in)) if ds_x * \
                n_in * h_in_L3 * w_in > 1024 else '%d B' % (ds_x * n_in * h_in_L3 * w_in * 1 / 8.)
            y_tot_size_str = '%.2f KiB' % (1. / 1024. / 8. * (ds_y * ch_out_L3 * h_out_L3 * w_out)) if ds_y * \
                ch_out_L3 * h_out_L3 * w_out > 1024 else '%d B' % (ds_y * ch_out_L3 * h_out_L3 * w_out * 1 / 8.)
            tile_n_in, tile_n_out, tile_h_in, tile_h_out, tile_w_in, tile_w_out = n_in, n_out, h_in, h_out, w_in, w_out
            x_tile_str = '[%dx%dx%d]' % (tile_n_in, tile_h_in, tile_w_in)
            y_tile_str = '[%dx%dx%d]' % (ch_out_L3, tile_h_out, tile_w_out)
            x_size_str = "%.2f KiB" % (1. / 1024. / 8. * (ds_x * tile_n_in * tile_h_in * tile_w_in)) if ds_x * \
                tile_n_in * tile_h_in * tile_w_in > 1024 else '%d B' % (ds_x * tile_n_in * tile_h_in * tile_w_in * 1 / 8.)
            y_size_str = '%.2f KiB' % (1. / 1024. / 8. * (ds_y * ch_out_L3 * tile_h_out * tile_w_out)) if ds_y * \
                ch_out_L3 * tile_h_out * tile_w_out > 1024 else '%d B' % (ds_y * ch_out_L3 * tile_h_out * tile_w_out * 1 / 8.)
            y_no_str = '%d' % (max(math.ceil((n_out) / (tile_n_out)), 1) * max(math.ceil(
                (h_out_L3) / (tile_h_out)), 1) * max(math.ceil((w_out) / (tile_w_out)), 1))
            x_no_str = '%d' % (factor_h_in)
            L2_tiles_size = ds_x * tile_n_in * tile_h_in * tile_w_in / 8. * (1 + int(factor_h_in > 1)) + ds_y * ch_out_L3 * tile_h_out * tile_w_out / 8. * (1 + int(factor_h_out > 1))
            logging.debug(f"  Precisions. x = {ds_x} bit, y = {ds_y} bit")
            logging.debug("    L3 size:".ljust(18) + "x: " + x_tot_str.ljust(15) +
                          "y: " + y_tot_str.ljust(15))
            logging.debug("    L3 buff:".ljust(18) + "x: " + x_tot_size_str.ljust(15) +
                          "y: " + y_tot_size_str.ljust(15))
            logging.debug("    tiles L3-L2:".ljust(18) + "x: " + x_tile_str.ljust(15) +
                          "y: " + y_tile_str.ljust(15))
            logging.debug("    L2 buff:".ljust(18) + "x: " + x_size_str.ljust(15) +
                          "y: " + y_size_str.ljust(15))
            logging.debug("    no. tiles:".ljust(18) + "x: " + x_no_str.ljust(15) +
                          "y: " + y_no_str.ljust(15))
            logging.debug("    Total L2 occupation:".ljust(18) + str(L2_tiles_size).ljust(15))
            if factor_h_in > 1 and factor_h_out > 1:
                logging.debug("    Tiling Input Act. and Output Act.")
            elif factor_h_in > 1:
                logging.debug("    Tiling Input Act.")
            elif factor_h_out > 1:
                logging.debug("    Tiling Output Act.")
        else:
            logging.debug("  No L3 tiling")
        # tiling of L2-L1. It is either the only tiling problem, or the first one for different L2 dimensions (top, middle, bottom)
        if factor_h_in > 1:
            h_out = int(np.floor((h_in - (fs1 - 1) + (s - 1)) / s))
        if factor_h_out > 1:
            h_in = h_out * s + (fs1 - 1) - (s - 1)
        tiling = self.get_tiling_pool2d_like(
            fs1,
            fs2,
            s,
            p_top,p_bottom,p_left,p_right,
            BN,
            n_in,
            n_out,
            [n_in, h_in, w_in],
            [n_out, h_out, w_out],
            self.buffer_size,
            name=name)       
        name_include.append(name)
        # report
        if tiling is not None:

            tile_n, tile_n, tile_h_in, tile_h_out, tile_w_in, tile_w_out = tiling
            x_tot_str = '[%dx%dx%d]' % (n_in, h_in, w_in)
            y_tot_str = '[%dx%dx%d]' % (n_out, h_out, w_out)
            x_tot_size_str = "%.2f KiB" % (1. / 1024. * (ds_x * n_in * h_in * w_in / 8.)) if ds_x * n_in * h_in * w_in > 1024 else '%d B' % (ds_x * n_in * h_in * w_in / 8.)
            y_tot_size_str = '%.2f KiB' % (1. / 1024. * (ds_y * n_out * h_out * w_out / 8.)) if ds_y * n_out * h_out * w_out > 1024 else '%d B' % (ds_y * n_out * h_out * w_out / 8.)

            x_tile_str = '[%dx%dx%d]' % (tile_n, tile_h_in, tile_w_in)
            y_tile_str = '[%dx%dx%d]' % (tile_n, tile_h_out, tile_w_out)
            x_size_str = "%.2f KiB" % (1. / 1024. * (ds_x * tile_n * tile_h_in * tile_w_in / 8.)) if ds_x * \
                tile_n * tile_h_in * tile_w_in > 1024 else '%d B' % (ds_x * tile_n * tile_h_in * tile_w_in / 8.)
            y_size_str = "%.2f KiB" % (1. / 1024. * (ds_y * tile_n * tile_h_out * tile_w_out / 8.)) if ds_y * \
                tile_n * tile_h_out * tile_w_out > 1024 else '%d B' % (ds_y * tile_n * tile_h_out * tile_w_out / 8.)

            x_no_str = '%d' % (max(math.ceil((n_in - tile_n) / (tile_n) + 1), 1) * max(math.ceil((h_in - tile_h_in) / (
                tile_h_in - fs1 + 1) + 1), 1) * max(math.ceil((w_in - tile_w_in) / (tile_w_in - fs2 + 1) + 1), 1))
            y_no_str = '%d' % (max(math.ceil((n_out) / (tile_n)), 1) * max(math.ceil(
                (h_out) / (tile_h_out)), 1) * max(math.ceil((w_out) / (tile_w_out)), 1))

            L1_tiles_size = ds_x * tile_n * tile_h_in * tile_w_in / 8. * (1 + int(int(x_no_str) > 1)) + ds_y * tile_n * tile_h_out * tile_w_out / 8. * (1 + int(int(y_no_str) > 1)) + n_out * 8
            logging.debug("    L2 size:".ljust(18) + "x: " + x_tot_str.ljust(15) +
                          "y: " + y_tot_str.ljust(15))
            logging.debug("    L2 buff:".ljust(18) + "x: " + x_tot_size_str.ljust(15) +
                          "y: " + y_tot_size_str.ljust(15))
            logging.debug("    tiles L2-L1:".ljust(18) + "x: " + x_tile_str.ljust(15) +
                          "y: " + y_tile_str.ljust(15))
            logging.debug("    L1 buff:".ljust(18) + "x: " + x_size_str.ljust(15) +
                          "y: " + y_size_str.ljust(15))
            logging.debug("    no. tiles:".ljust(18) + "x: " + x_no_str.ljust(15) +
                          "y: " + y_no_str.ljust(15))
            logging.debug("    Total L1 occupation:".ljust(18) + str(L1_tiles_size * 1.).ljust(15))
            # printing layer .c file. Either a unique one, or top,bottom and middle one (for which also tiling is computed).
            if (p_top+p_bottom) > 0 and (factor_h_in > 1 or factor_h_out > 1):
                in_dim1, out_dim1, weight_dim1, l2_dim_k, l2_dim_lambda, bias_dim1, l1_dim1, n_out1, w_out1, h_out1 = print_template_layer(
                    X, Y, W, n_in, h_in, w_in,
                    n_out, h_out, w_out,
                    tile_n, tile_h_in, tile_w_in,  tile_h_out, tile_w_out,
                    tile_n,
                    ds_x, ds_y, 0, 0, type_data,
                    fs1, fs2, 0, 0, p_left, p_right, s,
                    relu, 0, 0, out_mul, 0, out_shift,  factor_ch_out, factor_h_out, factor_h_in,
                    name_layer=name,
                    test=False,
                    test_location=self.test_location,
                    has_bias=True,
                    conv_order='PULP-NN-MAX',
                    optional=type,
                    l1_buffer=self.buffer_size,
                    platform=self.platform,
                    chip=self.chip,
                    optional_type=self.optional_type,
                    L3_tiling = L3_tiling,
                    sdk = self.sdk,
                    backend = self.backend,
                    number_of_clusters = self.number_of_clusters,
                    dma_parallelization = self.dma_parallelization)
            else:
                in_dim1, out_dim1, weight_dim1, l2_dim_k, l2_dim_lambda, bias_dim1, l1_dim1, n_out1, w_out1, h_out1 = print_template_layer(
                    X, Y, W, n_in, h_in, w_in,
                    n_out, h_out, w_out,
                    tile_n, tile_h_in, tile_w_in,  tile_h_out, tile_w_out,
                    tile_n,
                    ds_x, ds_y, 0, 0, type_data,
                    fs1, fs2, p_top, p_bottom, p_left, p_right, s,
                    relu, 0, 0, out_mul, 0, out_shift,  factor_ch_out, factor_h_out, factor_h_in,
                    name_layer=name,
                    test=False,
                    test_location=self.test_location,
                    has_bias=True,
                    conv_order='PULP-NN-MAX',
                    optional=type,
                    l1_buffer=self.buffer_size,
                    platform=self.platform,
                    chip=self.chip,
                    optional_type=self.optional_type,
                    L3_tiling = L3_tiling,
                    sdk = self.sdk,
                    backend = self.backend,
                    number_of_clusters = self.number_of_clusters,
                    dma_parallelization = self.dma_parallelization)  
            if (p_top + p_bottom) > 0 and (factor_h_in > 1 or factor_h_out > 1):
                tiling = self.get_tiling_pool2d_like(
                    fs1,
                    fs2,
                    s,
                    p_top,0,p_left,p_right,
                    BN,
                    n_in,
                    n_out,
                    [n_in, h_in, w_in],
                    [n_out, h_out, w_out],
                    self.buffer_size,
                    name=name)      
                tile_n_in, tile_n_out, tile_h_in, tile_h_out, tile_w_in, tile_w_out = tiling 
                in_dim1, out_dim1, weight_dim1, l2_dim_k, l2_dim_lambda, bias_dim1, l1_dim1, n_out1, w_out1, h_out1 = print_template_layer(
                    X, Y, W, n_in, h_in, w_in,
                    n_out, h_out, w_out,
                    tile_n, tile_h_in, tile_w_in,  tile_h_out, tile_w_out,
                    tile_n,
                    ds_x, ds_y, 0, 0, type_data,
                    fs1, fs2, p_top, 0, p_left, p_right, s,
                    relu, 0, 0, out_mul, 0, out_shift,  factor_ch_out, factor_h_out, factor_h_in,
                    name_layer=name,
                    test=False,
                    test_location=self.test_location,
                    has_bias=True,
                    conv_order='PULP-NN-MAX',
                    optional=type,
                    l1_buffer=self.buffer_size,
                    platform=self.platform,
                    chip=self.chip,
                    optional_type=self.optional_type,
                    L3_tiling = L3_tiling,
                    sdk = self.sdk,
                    backend = self.backend,
                    number_of_clusters = self.number_of_clusters,
                    dma_parallelization = self.dma_parallelization) 
                h_in_last = h_in
                #### CHECK WELL especially second nested if
                if factor_h_in > 2 or factor_h_out > 2:
                    if ((self.x_shape[-2] - h_in - h_in + conv_overlap_h + p_top) % (h_in - conv_overlap_h )) != 0:
                        h_in_last = ((self.x_shape[-2] - h_in - h_in + conv_overlap_h + p_top) % (h_in - conv_overlap_h )) + conv_overlap_h
                    pad_bot = p_bottom - ((self.x_shape[-2] - h_in - h_in + conv_overlap_h + p_top + p_bottom) % (h_in - conv_overlap_h ))
                elif factor_h_in > 1 or factor_h_out > 1:
                    if ((self.x_shape[-2] - h_in) % (h_in - conv_overlap_h -p_top)) != 0:
                        h_in_last = ((self.x_shape[-2] - h_in) % (h_in - conv_overlap_h -p_top)) + conv_overlap_h + p_bottom
                    pad_bot = p_bottom - ((self.x_shape[-2] - h_in) % (h_in - conv_overlap_h -p_top))
                tiling = self.get_tiling_pool2d_like(
                    fs1,
                    fs2,
                    s,
                    0,pad_bot,p_left,p_right,
                    BN,
                    n_in,
                    n_out,
                    [n_in, h_in, w_in],
                    [n_out, h_out, w_out],
                    self.buffer_size,
                    name=name)   
                tile_n_in, tile_n_out, tile_h_in, tile_h_out, tile_w_in, tile_w_out = tiling 
                in_dim1, out_dim1, weight_dim1, l2_dim_k, l2_dim_lambda, bias_dim1, l1_dim1, n_out1, w_out1, h_out1 = print_template_layer(
                    X, Y, W, n_in, h_in, w_in,
                    n_out, h_out, w_out,
                    tile_n, tile_h_in, tile_w_in,  tile_h_out, tile_w_out,
                    tile_n,
                    ds_x, ds_y, 0, 0, type_data,
                    fs1, fs2, 0, p_bottom, p_left, p_right, s,
                    relu, 0, 0, out_mul, 0, out_shift,  factor_ch_out, factor_h_out, factor_h_in,
                    name_layer=name,
                    test=False,
                    test_location=self.test_location,
                    has_bias=True,
                    conv_order='PULP-NN-MAX',
                    optional=type,
                    l1_buffer=self.buffer_size,
                    platform=self.platform,
                    chip=self.chip,
                    optional_type=self.optional_type,
                    L3_tiling = L3_tiling,
                    sdk = self.sdk,
                    backend = self.backend,
                    number_of_clusters = self.number_of_clusters,
                    dma_parallelization = self.dma_parallelization) 
                name_include.append(name + '_p_t')
                name_include.append(name + '_p_b')                   
            if self.test_location == 'L3_partial':
                full_net = 0
            else:
                full_net = 1 
            # print template layer for L3 execution of the layer, if present.
            if L3_tiling == 1 or input_L3 == 1:
                print_pool_template_layer_L3(
                    X, W, Y, fs1, fs2, p_top, s,
                    factor_ch_out, 
                    factor_h_out, 
                    factor_h_in,
                    name_include,
                    n_out * w_out * h_out,
                    n_in * w_in * h_in,
                    n_in * self.x_shape[-2] * self.x_shape[-1],
                    w_out,
                    h_out,
                    n_out,
                    w_in,
                    h_in,
                    n_in,
                    full_net,
                    self.platform,
                    ds_x,
                    ds_y,
                    self.test_location,
                    self.buffer_size,
                    input_L3,
                    backend = self.backend)
            ### L2 memory calculation
            if factor_h_out > 1:
                out_dim1 = out_dim1*2
            else:
                n_out_temp = self.out_ch
                h_in_temp = self.x_shape[-2]
                w_in_temp = self.x_shape[-1]
                h_out_temp = int(np.floor((h_in_temp - (fs1 - 1) + p_top + p_bottom + (s - 1)) / s))
                w_out_temp = int(np.floor((w_in_temp - (fs2 - 1) + p_left + p_right + (s - 1)) / s))
                out_dim1 = n_out_temp * h_out_temp * w_out_temp
            if factor_h_in > 1:
                in_dim1 = in_dim1*2
            else:
                n_in_temp = self.x_shape[0]
                h_in_temp = self.x_shape[-2]
                w_in_temp = self.x_shape[-1]
                in_dim1 = n_in_temp * h_in_temp * w_in_temp
            return in_dim1, out_dim1, 0, l1_dim1, L3_tiling, 1, factor_h_out, factor_h_in
        return None



    def get_tiling_pool2d_like(self,
                               filter_size1,
                               filter_size2,
                               stride,
                               p_top,p_bottom,p_left,p_right,
                               BN,
                               in_channels,
                               out_channels,
                               x_shape,
                               y_shape,
                               buffer_size,
                               name='pool'): 
        # This function generate the layer function to be included in the project for the pooling operation.
        parameters = pywrapcp.Solver.DefaultSolverParameters()
        solver = pywrapcp.Solver("simple_CP", parameters)
        cost_w = 10
        cost_h = 1
        cost_n = 10000
        cost_dim = 10000
        fs1 = filter_size1
        fs2 = filter_size2
        s = stride
        n_in = in_channels
        n_out = out_channels
        h_in = x_shape[-2] + p_top + p_bottom
        w_in = x_shape[-1] + p_left + p_right
        h_out = y_shape[-2]
        w_out = y_shape[-1]
        h_in = x_shape[-2]
        w_in = x_shape[-1]
        max_tile_n_out = n_out
        max_tile_n_in = n_in
        min_tile_w_in = fs2
        min_tile_h_in = fs1
        min_tile_w_out = 1
        min_tile_h_out = 1
        # this is to renormalize all costs
        max_obj_value = sys.maxsize
        memory = self.BitIn * n_in * h_in * w_in + self.BitOut * n_out * h_out * w_out + 4 * 8 * h_out * w_out
        if self.backend == 'Occamy':
            memory += self.BitIn * n_in * ((p_top + p_bottom) * (w_in + p_left + p_right) + (p_left + p_right) * h_in)
        if memory <= self.buffer_size * 8:
            db = 1
            return (n_in, n_out, h_in, h_out, w_in, w_out)
        else:
            db = 2
        # integer positive variables.
        tile_n = solver.IntVar(1, max_tile_n_in, 'tile_n')
        tile_h_in = solver.IntVar(min_tile_h_in, h_in, 'tile_h_in')
        tile_w_in = solver.IntVar(min_tile_w_in, w_in, 'tile_w_in')
        tile_h_out = solver.IntVar(min_tile_h_out, h_out, 'tile_h_out')
        tile_w_out = solver.IntVar(min_tile_w_out, w_out, 'tile_w_out')
        # scaling is used to ensure datasize is integer
        ds_x_scale = int(math.floor(32 * self.BitIn))
        ds_y_scale = int(math.floor(32 * self.BitOut))

        # CONSTRAINTS: managing of correct dimensions (no decimal h_out and any
        # type of rounding)
        # solver.Add(n_in == tile_n)
        solver.Add(0 == (tile_h_in - fs1) % s)
        solver.Add(0 == (tile_w_in - fs2) % s)
        solver.Add(solver.Max((h_in - tile_h_in - (tile_h_in - fs1 + 1 - p_top)), 0) % (tile_h_in - fs1 + 1) + abs(solver.Min(solver.Max((h_in - tile_h_in - (tile_h_in - fs1 + 1 - p_top)), 0) % (tile_h_in - fs1 + 1), 1) - 1) * fs1 >= fs1)
        solver.Add(solver.Max((w_in - tile_w_in - (tile_w_in - fs2 + 1 - p_left)), 0) % (tile_w_in - fs2 + 1) + abs(solver.Min(solver.Max((w_in - tile_w_in - (tile_w_in - fs2 + 1 - p_left)), 0) % (tile_w_in - fs2 + 1), 1) - 1) * fs2 >= fs2)
        constraint_all = db * ds_x_scale * tile_n * tile_h_in * tile_w_in + db * ds_y_scale * tile_n * tile_h_out * tile_w_out + 4 * ds_y_scale * tile_h_out * tile_w_out
        if self.backend == 'Occamy':
            constraint_all += db * ds_x_scale * tile_n * ((p_top + p_bottom) * (tile_w_in + p_left + p_right) + (p_left + p_right) * tile_h_in)
        solver.Add(constraint_all <= 32 * self.buffer_size * 8)
        if memory <= self.buffer_size * 8:
            solver.Add(tile_h_out * s == (tile_h_in - (fs1 - 1) + p_top + p_bottom + (s - 1)))
            solver.Add(tile_w_out * s == (tile_w_in - (fs2 - 1) + p_left + p_right + (s - 1)))
        else:
            solver.Add(tile_h_out * s == (tile_h_in - (fs1 - 1) + (s - 1)))
            solver.Add(tile_w_out * s == (tile_w_in - (fs2 - 1) + (s - 1)))
        # objective
        obj_expr = solver.IntVar(0, max_obj_value, "obj_expr")

        solver.Add(obj_expr == cost_dim * (ds_x_scale * tile_n * tile_h_in * tile_w_in + ds_y_scale * tile_n * tile_h_out * tile_w_out)
                   + 32 * cost_w * tile_w_in
                   + 32 * cost_h * tile_h_in
                   + 32 * cost_n * tile_n)
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
            if tile_h_in >= h_in:
                tile_h_in = h_in
                tile_h_out = int((tile_h_in -(fs1 - 1) + (p_top + p_bottom) + (s - 1))/s)
            if tile_w_in >= w_in:
                tile_w_in = w_in
                tile_w_out = int((tile_w_in -(fs2 - 1) + (p_left + p_right) + (s - 1))/s)
            return (tile_n, tile_n, tile_h_in, tile_h_out, tile_w_in, tile_w_out)
        print("  Pool2d ERROR: no tiling found. Exiting...")
        os._exit(0) 
        return None

    def get_tiling_Add(self, X, Y, W,
                       cost_dim=10000,
                       cost_w=100,
                       cost_n=10,
                       cost_h=1,
                       max_tile_n_in=None,
                       max_tile_n_out=None,
                       min_tile_w_in=None,
                       min_tile_h_in=None,
                       min_tile_w_out=None,
                       min_tile_h_out=None,
                       type_data='char',
                       multiple_buffering_factor=2,
                       ds_x=1,
                       ds_y=1,
                       ds_W=1,
                       relu=0,
                       out_mul1=0,
                       out_mul2=0,
                       out_shift=0,
                       name='Add',
                       type='Avg'
                       ):
        # This function generate the layer function to be included in the project for the addition operation.

        parameters = pywrapcp.Solver.DefaultSolverParameters()
        solver = pywrapcp.Solver("simple_CP", parameters)
        cost_w = 10
        cost_h = 1
        cost_n = 1000
        cost_dim = 1000000

        n_in = n_out = self.x_shape[0]
        h_in = self.x_shape[-2]
        w_in = self.x_shape[-1]
        h_out = h_in
        w_out = w_in
        
        ds_x = self.BitIn
        ds_y = self.BitOut
        if max_tile_n_out is None:
            max_tile_n_out = n_out
        if max_tile_n_in is None:
            max_tile_n_in = n_in
        if min_tile_w_out is None:
            min_tile_w_out = 1
        if min_tile_h_out is None:
            min_tile_h_out = 1
        # this is to renormalize all costs
        max_obj_value = sys.maxsize
        memory = ds_x * n_in * h_in * w_in * 2 + ds_y * n_out * h_out * w_out
        if memory >= self.L2_buffer_size * 8:
            print("  Add ERROR: no tiling from L3 supported. Exiting...")
            os._exit(0)            
        if memory <= self.buffer_size * 8:
            db_x = 1
            db_y = 1
            db_W = 1
        else:
            db_x = multiple_buffering_factor
            db_y = multiple_buffering_factor
            db_W = multiple_buffering_factor
        # integer positive variables.
        tile_n = solver.IntVar(1, max_tile_n_in, 'tile_n')
        tile_h_in = solver.IntVar(1, h_in, 'tile_h_in')
        tile_w_in = solver.IntVar(1, w_in, 'tile_w_in')
        tile_h_out = solver.IntVar(min_tile_h_out, h_out, 'tile_h_out')
        tile_w_out = solver.IntVar(min_tile_w_out, w_out, 'tile_w_out')

        # scaling is used to ensure datasize is integer
        ds_x_scale = int(math.ceil(32 * ds_x))
        ds_y_scale = int(math.ceil(32 * ds_y))
        solver.Add(tile_h_out == tile_h_in)
        solver.Add(tile_w_out == tile_w_in)            
        solver.Add(tile_n == n_in)

        # CONSTRAINTS: managing of correct dimensions (no decimal h_out and any
        # type of rounding)
        solver.Add(db_x * ds_x_scale * tile_n * tile_h_in * tile_w_in * 2 + db_y * ds_y_scale * tile_n * tile_h_out * tile_w_out <= 32 * self.buffer_size * 8)
        # objective
        obj_expr = solver.IntVar(0, max_obj_value, "obj_expr")
        solver.Add(obj_expr == cost_dim * (ds_x_scale * tile_n * tile_h_in * tile_w_in + ds_y_scale * tile_n * tile_h_out * tile_w_out)
                   + 32 * cost_w * tile_w_in
                   + 32 * cost_h * tile_h_in
                   + 32 * cost_n * tile_n)
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

            x_tot_str = '[%dx%dx%d]' % (n_in, h_in, w_in)
            y_tot_str = '[%dx%dx%d]' % (n_out, h_out, w_out)
            x_tot_size_str = "%.2f KiB" % (1. / 1024. * (ds_x * n_in * h_in * w_in / 8.)) if ds_x * \
                n_in * h_in * w_in > 1024 else '%d B' % (ds_x * n_in * h_in * w_in / 8.)
            y_tot_size_str = '%.2f KiB' % (1. / 1024. * (ds_y * n_out * h_out * w_out / 8.)) if ds_y * \
                n_out * h_out * w_out > 1024 else '%d B' % (ds_y * n_out * h_out * w_out / 8.)

            x_tile_str = '[%dx%dx%d]' % (tile_n, tile_h_in, tile_w_in)
            y_tile_str = '[%dx%dx%d]' % (tile_n, tile_h_out, tile_w_out)

            x_size_str = "%.2f KiB" % (1. / 1024. * (ds_x * tile_n * tile_h_in * tile_w_in / 8.)) if ds_x * \
                tile_n * tile_h_in * tile_w_in > 1024 else '%d B' % (ds_x * tile_n * tile_h_in * tile_w_in / 8.)
            y_size_str = "%.2f KiB" % (1. / 1024. * (ds_y * tile_n * tile_h_out * tile_w_out / 8.)) if ds_y * \
                tile_n * tile_h_out * tile_w_out > 1024 else '%d B' % (ds_y * tile_n * tile_h_out * tile_w_out / 8.)

            x_no_str = '%d' % (max(math.ceil((n_in - tile_n) / (tile_n) + 1), 1) * max(math.ceil(
                (h_in - tile_h_in) / (tile_h_in) + 1), 1) * max(math.ceil((w_in - tile_w_in) / (tile_w_in) + 1), 1))
            y_no_str = '%d' % (max(math.ceil((n_out) / (tile_n)), 1) * max(math.ceil(
                (h_out) / (tile_h_out)), 1) * max(math.ceil((w_out) / (tile_w_out)), 1))

            logging.debug("  Add tiling:")
            logging.debug("    L2/L3 size:".ljust(18) + "x: " +
                          x_tot_str.ljust(15) + "y: " + y_tot_str.ljust(15))
            logging.debug("    L2/L3 buf:".ljust(18) + "x: " +
                          x_tot_size_str.ljust(15) + "y: " + y_tot_size_str.ljust(15))
            logging.debug("    tiles:".ljust(18) + "x: " +
                          x_tile_str.ljust(15) + "y: " + y_tile_str.ljust(15))
            logging.debug("    buffers:".ljust(18) + "x: " +
                          x_size_str.ljust(15) + "y: " + y_size_str.ljust(15))
            logging.debug("    no. tiles:".ljust(18) + "x: " +
                          x_no_str.ljust(15) + "y: " + y_no_str.ljust(15))
            in_dim1, out_dim1, weight_dim1,  l2_dim_k, l2_dim_lambda, bias_dim1, l1_dim1, n_out1, w_out1, h_out1 = print_template_layer(
                X, Y, W, n_in, h_in, w_in,
                n_out, h_out, w_out,
                tile_n, tile_h_in, tile_w_in,  tile_h_out, tile_w_out,
                tile_n,
                ds_x, ds_y, 0, 0, type_data,
                1, 1, 0,0,0,0, 1,
                relu, 0, 0, out_mul1, out_mul2, out_shift, 1, 1, 1,
                name_layer=name,
                test=False,
                test_location=self.test_location,
                has_bias=True,
                conv_order='PULP-NN-ADD',
                optional='add',
                l1_buffer=self.buffer_size,
                platform=self.platform,
                chip=self.chip,
                optional_type=self.optional_type,
                sdk = self.sdk,
                backend = self.backend,
                number_of_clusters = self.number_of_clusters,
                dma_parallelization = self.dma_parallelization)
            return in_dim1, out_dim1, 0, l1_dim1, 0, 1, 1, 1
        print("  Add ERROR: no tiling found. Exiting...")
        os._exit(0)
        return None
