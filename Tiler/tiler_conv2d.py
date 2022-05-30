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

class Tiler_Conv2D():
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
        if n_in < self.number_of_clusters:
            input_dim = self.BitIn * n_in * h_in * w_in
        else:
            input_dim = self.BitIn * int(n_in/self.number_of_clusters) * h_in * w_in
        if self.backend == 'Occamy':
            if n_in < self.number_of_clusters:
                input_dim = self.BitIn * n_in * (h_in + padding_top + padding_bottom) * (w_in + padding_left + padding_right)
                input_dim += ((padding_top + padding_bottom) * (w_in + padding_left + padding_right) + (padding_left + padding_right) * h_in) * self.BitIn * n_in
            else:
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
            if h_in <= 32 and w_in <= 32 and self.backend != 'Occamy':
                solver.Add(tile_h_in == h_in)
                solver.Add(tile_w_in == w_in)
                solver.Add(tile_h_out == h_out)
                solver.Add(tile_w_out == w_out)
            elif h_in > 32 or w_in > 32:
                solver.Add(tile_h_out * s == (tile_h_in - (fs1 - 1) + ((tile_h_in % h_in) == 0) * (padding_top + padding_bottom) + (s - 1)))
                #solver.Add(tile_w_out * s == (tile_w_in - (fs2 - 1) + ((tile_w_in % w_in) == 0) * (padding_left + padding_right) + (s - 1)))
                solver.Add(tile_w_in == w_in)
                solver.Add(tile_w_out == w_out)
            if self.backend == 'Occamy':
                solver.Add(tile_h_out * s == (tile_h_in - (fs1 - 1) + ((tile_h_in % h_in) == 0) * (padding_top + padding_bottom) + (s - 1)))
                solver.Add(tile_w_in == w_in)
                solver.Add(tile_w_out == w_out)
                solver.Add(tile_n_out % 2 == 0)
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
            constr_weight = db * ds_W_scale * tile_n_in * fs1 * fs2
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
                            + 64 * 10000 * ((n_out-zero_variable-1) % (tile_n_out))
                ####### Geometrical Shape of Border Tiles #####
                heuristics += 64 * 10000 * ((n_out-zero_variable-1) % (tile_n_out)) \
                            + 64 * 10000 * (((n_out-zero_variable-1) % (tile_n_out)) % 4) \
                            + 64 * 20000 * (((h_out-zero_variable-1) % (tile_h_out)) % 8) \
                            + 64 * 30000 * (((w_out-zero_variable-1) % (tile_w_out)) % 2)
            elif DW == 1:
                ####### Geometrical Shape of Tiles ############
                heuristics += 32 * 10000 * ((tile_n_out > 7)) \
                            + 64 * 10000 * ((tile_n_out - 1) % 16) \
                            + 32 * 10000 * ((tile_h_out % 4) == 0)
                ####### Total Dimension of Tile ###############
                heuristics += constraint_all
                ####### Maximization of Reuse of im2col #######
                heuristics += 32 * 1000 * tile_w_out \
                            + 32 * 1000 * tile_h_out \
                            + 32 * 100 * (((h_out-zero_variable-1) % (tile_h_out))) \
                            + 32 * 100 * (((w_out-zero_variable-1) % (tile_w_out)))
                ####### Geometrical Shape of Border Tiles #####
                heuristics += 32 * 100 * (((n_out-zero_variable-1) % (tile_n_out)) > 7) \
                            + 32 * 100 * (((h_out-zero_variable-1) % (tile_h_out)) % 4)
        else:
            ####### Geometrical Shape of Tiles ############
            heuristics += 64 * 10000 * (((((h_out-zero_variable - 1) % tile_h_out)) % 8) > 4) 
            heuristics += 64 * 10000 * (((tile_h_out - 1) % 8) > 4) \
                        + 64 * 10000 * ((tile_n_out - 1) % 8) \
                        + 64 * 10000 * (tile_w_out * tile_h_out) ## better to have a bigger spatial dimension then more out_channels to reduce memory overhead in copying overlapping input pixels
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

