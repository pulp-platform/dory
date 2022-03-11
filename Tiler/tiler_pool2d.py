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
from Layer2D_templates_writer import print_template_layer
from Layer1D_templates_writer import print_template_layer_1D
from L3_templates_writer import print_template_layer_L3
from L3_templates_writer import print_pool_template_layer_L3
import logging
import os
import sys

class Tiler_Pool2D():
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
