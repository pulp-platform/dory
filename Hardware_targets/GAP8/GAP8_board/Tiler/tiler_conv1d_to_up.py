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

# constraint solver for optimization
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import solver_parameters_pb2

# template for output
from Layer2D_templates_writer import print_template_layer
from Layer1D_templates_writer import print_template_layer_1D
from L3_templates_writer import print_template_layer_L3
from L3_templates_writer import print_pool_template_layer_L3
import logging
import os
import sys

class Tiler_Conv1D():
    # Class to generate the Tiling of the layer.
    def __init__(self,tiler):
        self.module = tiler.module
        self.out_ch = tiler.out_ch
        self.filter_size = tiler.filter_size
        self.stride = tiler.stride
        self.padding = tiler.padding
        self.groups = tiler.groups
        self.x_shape = tiler.x_shape
        self.buffer_size = tiler.buffer_size
        self.L2_buffer_size = tiler.L2_buffer_size
        self.platform = tiler.platform
        self.chip = tiler.chip
        self.test_location = tiler.test_location
        self.BitIn = tiler.BitIn
        self.BitW = tiler.BitW
        self.BitOut = tiler.BitOut
        self.BitActivation = tiler.BitActivation
        self.optional_type = tiler.optional_type
        self.sdk = tiler.sdk
        self.backend = tiler.backend
        self.dma_parallelization = tiler.dma_parallelization
        self.number_of_clusters = tiler.number_of_clusters

    def get_tiling(self, X, Y, W,
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

