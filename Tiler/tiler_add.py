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

class Tiler_Add():
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
