# -*- coding: future_fstrings -*-     # should work even without -*-
# -*- coding: utf-8 -*-
# Model_deployment.py
# Alessio Burrello <alessio.burrello@unibo.it>
# Thorir Mar Ingolfsson <thoriri@iis.ee.ethz.ch>
#
# Copyright (C) 2019-2020 University of Bologna
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

import torch
import numpy as np
from tiling import Tiling
import template as template
import os
import pandas as pd
from mako.template import Template
from collections import OrderedDict
import logging

class Model_deployment():
    """
    Used to manage the PULP graph. By now, supported Convolutions, Pooling, Linear Layers and Relu.
    """

    def __init__(self, platform, chip):
        self.platform = platform
        self.chip = chip

    def copy_files(self, optional, layer_mixed_list,version, sdk, backend, dma_parallelization):
        print("The function copy_files should be implemented in the target Backend. Exiting ...")

    def copy_backend(self, BitActivation, PULP_Nodes_Graph, number_of_deployed_layers, sdk, backend, dma_parallelization):
        print("The function copy_backend should be implemented in the target Backend. Exiting ...")
        exit(0)

    def create_weights_files(self, PULP_Nodes_Graph, number_of_deployed_layers, BitActivation, load_dir):
        print("The function create_weights_files should be implemented in the target Backend. Exiting ...")
        exit(0)


    def create_layers_tiling(self, PULP_Nodes_Graph, 
                            number_of_deployed_layers, 
                            L1_dimension,
                            l2_buffer_size, 
                            BitActivation, 
                            performance_single_layer, 
                            sdk,
                            backend,
                            dma_parallelization,
                            number_of_clusters,
                            type_data = 'float'):
        ####################################################################################
        ###### SECTION 3: PARSING OF EACH LAYER INDEPENDENT. TILING + LAYER CREATION  ######
        ####################################################################################
        name_list = []
        layer_list = []
        stringa_features = []
        name_layer_list = []
        name_layer_list_internal = []       
        MAC_total = 0
        Layers_L3_input_act = 0
        Layers_L3_output_act = 0
        Layers_L3_weights = 0
        L2_memory_occupation = 0
        factor_h_out = 1
        optional = '8bit'
        for i, nodes_to_deploy in enumerate(PULP_Nodes_Graph[:number_of_deployed_layers]):
            if nodes_to_deploy.get_parameter('out_activation_bits') < 8 or nodes_to_deploy.get_parameter('input_activation_bits') < 8 or nodes_to_deploy.get_parameter('weight_bits') < 8:
                optional = 'mixed-sw'
        for i, nodes_to_deploy in enumerate(PULP_Nodes_Graph[:number_of_deployed_layers]):
            if('Conv' in nodes_to_deploy.name or 'Gemm' in nodes_to_deploy.name or 'MatMul' in nodes_to_deploy.name):
                layer = 'Conv'
                if 'Conv' in nodes_to_deploy.name:
                    h_dimension = nodes_to_deploy.get_parameter('kernel_shape')[0] + nodes_to_deploy.get_parameter('input_dim')[0] + nodes_to_deploy.get_parameter('output_dim')[0]
                    if h_dimension == 3:
                        layer = 'Conv1D'
                        optional = '1D_Conv'
            elif('Pool' in nodes_to_deploy.name):
                layer = 'Pool'
            elif('Add' in nodes_to_deploy.name):
                layer = 'Add'
            name_layer = "layer" + nodes_to_deploy.name + str(i)
            ######################## NEED A  FIX ####################################################
            #### OTHERWISE ONLY WEIGHT < L2/2 GO in L2 --> much more L3 tiling not needed############
            #########################################################################################
            tile_factor = 1.8
            if (i < len(PULP_Nodes_Graph)-1) and ('Conv' in PULP_Nodes_Graph[i+1].name or 'Gemm' in PULP_Nodes_Graph[i+1].name or 'MatMul' in PULP_Nodes_Graph[i+1].name):
                if PULP_Nodes_Graph[i+1].ch_in*PULP_Nodes_Graph[i+1].ch_out*PULP_Nodes_Graph[i+1].kernel_shape[0]*PULP_Nodes_Graph[i+1].kernel_shape[1] > int(l2_buffer_size/tile_factor):
                    weight_overhead = int(l2_buffer_size/tile_factor)
                else:
                    weight_overhead = int(PULP_Nodes_Graph[i+1].weight_bits*PULP_Nodes_Graph[i+1].ch_in*PULP_Nodes_Graph[i+1].ch_out*PULP_Nodes_Graph[i+1].kernel_shape[0]*PULP_Nodes_Graph[i+1].kernel_shape[1]/8) +int(PULP_Nodes_Graph[i+1].ch_out*BitActivation/8*2)
            else:
                weight_overhead = 0
            BitIn = PULP_Nodes_Graph[i].input_activation_bits
            BitOut = PULP_Nodes_Graph[i].out_activation_bits
            if 'weights' in PULP_Nodes_Graph[i].__dict__:
                BitW = PULP_Nodes_Graph[i].weight_bits
            if i == len(PULP_Nodes_Graph)-1:
                name_layer = name_layer + '_last'
            if(performance_single_layer == 'Yes'):
                test_location = 'L3+performance'
            else:
                test_location = 'L3'
            tile_gen = Tiling(layer,
                              nodes_to_deploy.ch_out,
                              nodes_to_deploy.kernel_shape,
                              nodes_to_deploy.strides,
                              nodes_to_deploy.pads,
                              nodes_to_deploy.group,
                              [nodes_to_deploy.ch_in * nodes_to_deploy.group,nodes_to_deploy.input_dim[0], nodes_to_deploy.input_dim[1]],
                              L1_dimension,
                              l2_buffer_size-weight_overhead,
                              self.platform,
                              self.chip,
                              test_location=test_location,
                              BitIn=BitIn,
                              BitW=BitW,
                              BitOut=BitOut,
                              BitActivation = BitActivation,
                              optional_type=optional,
                              sdk = sdk,
                              backend = backend,
                              dma_parallelization = dma_parallelization,
                              number_of_clusters = number_of_clusters)
            str_l = 'ch_in' + str(nodes_to_deploy.ch_in) + 'ch_out' + str(nodes_to_deploy.ch_out) + 'groups' + str(
                nodes_to_deploy.group) + 'dim_image' + str(nodes_to_deploy.input_dim[1],) + 'stride' + str(nodes_to_deploy.strides) + 'kernel'+ str(
                nodes_to_deploy.kernel_shape[0]) + str(nodes_to_deploy.kernel_shape[1]) + 'BitIn' + str(BitIn) + 'BitOut' + str(BitOut) + 'BitW' + str(BitW)
            if '1D' in layer:
                str_l += 'Dilation' + str(nodes_to_deploy.dilations)
            name = nodes_to_deploy.name
            for scan_i, _ in enumerate(stringa_features):
                if(str_l == stringa_features[scan_i] and str(layer) == str(layer_list[scan_i])):
                    name_layer = name_layer_list[scan_i]
                    name = name_layer_list_internal[scan_i]
            stringa_features.append(str_l)
            layer_list.append(layer)
            name_layer_list.append(name_layer)
            name_layer_list_internal.append(name)
            relu = 0
            BN = 0
            DW = 0
            input_dim_constraint = 0
            output_weights_dim_constraint = 0
            if(i == 0):
                weight_constraint = 0
            if(i == 0):
                input_L3 = 0
            elif(factor_h_out > 1):
                input_L3 = 1
                input_dim_constraint = out_dim2
                output_weights_dim_constraint = l2_buffer_size - weight_overhead - out_dim2_old
                if(output_weights_dim_constraint < 0):
                    print("Problems with current implementation on L3 tiling. Prediction of weights of next layer not accurate. Exiting...")
                    os._exit(0)
            else:
                input_L3 = 0
            if('Relu' in nodes_to_deploy.name):
                relu = 1
            if('BN' in nodes_to_deploy.name):
                BN = 1
            if('DW' in nodes_to_deploy.name):
                DW = 1
            ###### TO MODIFY ########
            if 'Relu' not in nodes_to_deploy.name:
                nodes_to_deploy.outmul = 1
                if 'Add' not in nodes_to_deploy.name:
                    nodes_to_deploy.outshift = 1
            if 'bias' in nodes_to_deploy.__dict__:
                h_b = 1
            else:
                h_b = 0
            if('Conv1D' in layer):
                d = dict(X=0, Y=0, W=0,
                        relu=relu, BN=BN,
                        type_data = type_data,
                        dilation=nodes_to_deploy.dilations,
                        has_bias=h_b,
                        out_mul=nodes_to_deploy.outmul,
                        out_shift=nodes_to_deploy.outshift,
                        name=name_layer)
            elif('Gemm' in nodes_to_deploy.name or 'Conv' in nodes_to_deploy.name or 'MatMul' in nodes_to_deploy.name):
                d = dict(X=0, Y=0, W=0,
                        relu=relu, BN=BN, DW=DW,
                        type_data = type_data,
                        has_bias=h_b,
                        out_mul=nodes_to_deploy.outmul,
                        out_shift=nodes_to_deploy.outshift,
                        name=name_layer,
                        input_L3 = input_L3,
                        input_dim_constraint = input_dim_constraint,
                        output_weights_dim_constraint = output_weights_dim_constraint,
                        weight_constraint = weight_constraint)
            elif('Pool' in nodes_to_deploy.name):
                d = dict(X=0, Y=0, W=0,
                        relu=relu, BN = BN,
                        type_data = type_data,
                        out_mul=nodes_to_deploy.outmul,
                        out_shift=nodes_to_deploy.outshift,
                        name=name_layer,
                        input_L3 = input_L3,
                        input_dim_constraint = input_dim_constraint,
                        output_weights_dim_constraint = output_weights_dim_constraint,
                        type=name)
            elif('Add' in nodes_to_deploy.name):
                d = dict(X=0, Y=0, W=0,
                        relu=relu,
                        type_data = type_data,
                        out_mul1=nodes_to_deploy.inmul1,
                        out_mul2=nodes_to_deploy.inmul2,
                        out_shift=nodes_to_deploy.outshift,
                        name=name_layer,
                        type=name)
            in_dim2, out_dim2, weights_dim, l1_dim2, L3_tiling, factor_ch_out, factor_h_out, factor_h_in = tile_gen.get_tiling(**d)
            if(factor_ch_out > 1):
                PULP_Nodes_Graph[i].L3_allocation = 1
            else:
                PULP_Nodes_Graph[i].L3_allocation = 0
            Layers_L3_input_act += int(factor_h_in > 1)
            Layers_L3_output_act += int(factor_h_out > 1)
            Layers_L3_weights += int(factor_ch_out > 1)
            PULP_Nodes_Graph[i].L3_input = int(factor_h_in > 1)
            PULP_Nodes_Graph[i].L3_output = int(factor_h_out > 1)
            PULP_Nodes_Graph[i].L3_weights = int(factor_ch_out > 1)
            if(i == 0):
                out_dim2_old = in_dim2
            if(factor_h_out > 1):
                out_dim2 = l2_buffer_size - weight_overhead - out_dim2_old - weights_dim
            out_dim2_old = out_dim2
            while weights_dim % 4 != 0:
                weights_dim += 1
            if(weight_overhead == int(l2_buffer_size/2)):
                weight_constraint = int(l2_buffer_size/2)
            else:
                weight_constraint = 0
            if(L3_tiling == 1):
                name_layer = name_layer + 'L3'
                try:
                    PULP_Nodes_Graph[i].input_activation_dimensions_L3 = int(PULP_Nodes_Graph[i].input_dim[0] * PULP_Nodes_Graph[i].input_dim[1] * PULP_Nodes_Graph[i].ch_in*BitIn/8)
                except:
                    PULP_Nodes_Graph[i].input_activation_dimensions_L3 = int(PULP_Nodes_Graph[i].input_dim * PULP_Nodes_Graph[i].ch_in*BitIn/8)
                try:
                    PULP_Nodes_Graph[i].output_activation_dimensions_L3 = int(PULP_Nodes_Graph[i].output_dim[0] * PULP_Nodes_Graph[i].output_dim[1] * PULP_Nodes_Graph[i].ch_out*BitOut/8)
                except:
                    PULP_Nodes_Graph[i].output_activation_dimensions_L3 = int(PULP_Nodes_Graph[i].output_dim * PULP_Nodes_Graph[i].ch_out*BitOut/8)
            name_list.append(name_layer)
            if('Gemm' in nodes_to_deploy.name or 'Conv' in nodes_to_deploy.name or 'MatMul' in nodes_to_deploy.name):
                if(i > 0):
                    PULP_Nodes_Graph[i].weights_dimension = PULP_Nodes_Graph[i-1].weights_dimension + weights_dim
                else:
                    PULP_Nodes_Graph[i].weights_dimension = weights_dim
            else:
                PULP_Nodes_Graph[i].weights_dimension = PULP_Nodes_Graph[i-1].weights_dimension
            if('Gemm' in nodes_to_deploy.name or 'Conv' in nodes_to_deploy.name or 'MatMul' in nodes_to_deploy.name):
                if(factor_ch_out == 1):
                    if(i > 0):
                        PULP_Nodes_Graph[i].weights_dimension_L3 = PULP_Nodes_Graph[i-1].weights_dimension_L3 + weights_dim
                    else:
                        PULP_Nodes_Graph[i].weights_dimension_L3 = weights_dim
                else:
                    if(i > 0):
                        PULP_Nodes_Graph[i].weights_dimension_L3 = PULP_Nodes_Graph[i-1].weights_dimension_L3 + int(weights_dim*factor_ch_out/2)
                    else:
                        PULP_Nodes_Graph[i].weights_dimension_L3 = int(weights_dim*factor_ch_out/2)                    
            else:
                PULP_Nodes_Graph[i].weights_dimension_L3 = PULP_Nodes_Graph[i-1].weights_dimension_L3
            PULP_Nodes_Graph[i].input_activation_dimensions = int(in_dim2*BitIn/8)
            PULP_Nodes_Graph[i].output_activation_dimensions = int(out_dim2*BitOut/8)
            if(i > 0):
                if(PULP_Nodes_Graph[i].input_activation_dimensions != PULP_Nodes_Graph[i-1].output_activation_dimensions) and PULP_Nodes_Graph[i-1].L3_output==1:
                    PULP_Nodes_Graph[i].input_activation_dimensions = PULP_Nodes_Graph[i-1].output_activation_dimensions
            PULP_Nodes_Graph[i].l1_dimensions = l1_dim2
            if('Pool' not in nodes_to_deploy.name):
                MAC_total += nodes_to_deploy.MACs
        return PULP_Nodes_Graph, Layers_L3_input_act, Layers_L3_output_act, Layers_L3_weights, name_layer_list, name_list, MAC_total

    def generate_intermediate_activations(self, PULP_Nodes_Graph, 
                                        load_dir, 
                                        number_of_deployed_layers, 
                                        check_layer,
                                        weights_to_write):
        ######################################################################################
        ###### SECTION 4: GENERATE CHECKSUM BY USING WEIGHT AND OUT_LAYER{i}.TXT FILES  ######
        ######################################################################################        
        try:
            x_in = pd.read_csv(load_dir + 'input.txt')
            x_in = x_in.values[:, 0].astype(int)
        except:
            x_in = torch.Tensor(1, PULP_Nodes_Graph[0].group, PULP_Nodes_Graph[0].ch_in, PULP_Nodes_Graph[0].input_dim[0], PULP_Nodes_Graph[0].input_dim[1]).uniform_(0, (2**(9)))
            x_in[x_in > (2**8 - 1)] = 0
            x_in = torch.round(x_in)
            x_in = x_in.flatten().numpy().astype(int)
        for i, _ in enumerate(x_in):
            x_in[i] = np.uint8(x_in[i])
        PULP_Nodes_Graph[0].check_sum_in = sum(x_in)
        f_w = 0
        for f, nodes_to_deploy in enumerate(PULP_Nodes_Graph[:number_of_deployed_layers]):
            X_in = pd.read_csv(load_dir + 'out_layer' + str(f) + '.txt')
            X_in = X_in.values[:, 0].astype(int)
            if f == len(PULP_Nodes_Graph[:number_of_deployed_layers]) - 1:
                class_out = np.where(X_in == np.max(X_in))[0][0]
            for i, _ in enumerate(X_in):
                X_in[i] = np.uint8(X_in[i])
            BitIn = nodes_to_deploy.input_activation_bits
            BitOut = nodes_to_deploy.out_activation_bits

            Input_compressed = []
            z = 0
            import copy
            Loop_over = copy.deepcopy(X_in)
            if f != len(PULP_Nodes_Graph[:number_of_deployed_layers]) - 1:
                for _, i_x in enumerate(Loop_over):
                    if (z % int(8 / BitOut)) == 0:
                        Input_compressed.append(int(i_x.item()))
                    else:
                        Input_compressed[-1] += int(i_x.item()) << (BitOut * (z % int(8 / BitOut)))
                    z += 1
            if check_layer == f:
                act_compare = Input_compressed
            PULP_Nodes_Graph[f].check_sum_out = sum(Input_compressed)
            if f == len(PULP_Nodes_Graph) - 1:
                ww = np.asarray(nodes_to_deploy.weights).reshape(nodes_to_deploy.ch_out,nodes_to_deploy.ch_in ).astype(np.int8).astype(int)
                X_in = pd.read_csv(load_dir + 'out_layer' + str(f-1) + '.txt')
                X_out = pd.read_csv(load_dir + 'out_layer' + str(f) + '.txt')
                X_in = X_in.values[:, 0].astype(int).reshape(X_in.shape[0],1)
                try:
                    PULP_Nodes_Graph[f].check_sum_out = sum(sum(np.matmul(ww,X_in)))
                except:
                    PULP_Nodes_Graph[f].check_sum_out = 0
            if f != len(PULP_Nodes_Graph[:number_of_deployed_layers]) - 1:
                PULP_Nodes_Graph[f + 1].check_sum_in = sum(Input_compressed)
            if 'Gemm' in nodes_to_deploy.name or 'Conv' in nodes_to_deploy.name or 'MatMul' in nodes_to_deploy.name:
                PULP_Nodes_Graph[f].check_sum_w = sum(weights_to_write[f_w])
                f_w += 1
            else:
                PULP_Nodes_Graph[f].check_sum_w = 0
        return PULP_Nodes_Graph, class_out

    def print_model_network(self, PULP_Nodes_Graph,
                            number_of_deployed_layers=29,
                            load_dir='./mnistNet/',
                            check_layer=0,
                            verbose_level='None',
                            performance_single_layer='Yes',
                            L1_dimension = 35000,
                            master_stack = 4096,
                            slave_stack = 3072,
                            l2_buffer_size = 400000,
                            fc_frequency = 100000000,
                            cl_frequency = 100000000,
                            BitActivation = 32,
                            sdk='gap_sdk', 
                            backend='MCU', 
                            dma_parallelization='8-cores',
                            number_of_clusters = 1,
                            type_data = 'char'):
        # Function used to create all the files for the application
        # copy backend is used to copy all the files of the backend
        self.copy_backend(BitActivation, PULP_Nodes_Graph, number_of_deployed_layers, sdk, backend, dma_parallelization)
        # create L3 files for weights. These files are .hex which are copied in hyperflash then
        PULP_Nodes_Graph, weights_files_list, weights_to_write = self.create_weights_files(PULP_Nodes_Graph, number_of_deployed_layers, BitActivation, load_dir)
        fileh = logging.FileHandler('logs/Tiling_profiling.log', 'a')
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fileh.setFormatter(formatter)
        fileh.setLevel(logging.DEBUG)
        log = logging.getLogger() 
        for hdlr in log.handlers[:]:
            log.removeHandler(hdlr)
        log.addHandler(fileh)
        print("Creating tiling profiling in Tiling_profling.log")
        # tiling of all the layers. Both tiling and layer generation
        PULP_Nodes_Graph, num_L3_input_tile, num_L3_output_tile, num_L3_weight_tile, name_layer_list, name_list, MAC_total = self.create_layers_tiling(PULP_Nodes_Graph, 
            number_of_deployed_layers, 
            L1_dimension, 
            l2_buffer_size, 
            BitActivation,              
            performance_single_layer,
            sdk,
            backend,
            dma_parallelization,
            number_of_clusters,
            type_data = type_data)

        logging.debug("  ")
        logging.debug("  Layers with L3 input activation: " + str(num_L3_input_tile))
        logging.debug("  Layers with L3 output activation: " + str(num_L3_output_tile))
        logging.debug("  Layers with L3 weights: " + str(num_L3_weight_tile))

        name_layer_list_unique = list(set(name_layer_list))
        for i, _ in enumerate(name_layer_list_unique):
            name_layer_list_unique[i] = name_layer_list_unique[i] + ".c"
        for i, nodes_to_deploy in enumerate(PULP_Nodes_Graph[:number_of_deployed_layers]):
            if nodes_to_deploy.L3_allocation == 1:
                name_layer_list_unique.append(name_layer_list[i] + "L3" + ".c")
        # compute the checksums for intermediate activations checking
        if 'Check' in verbose_level or 'Last' in verbose_level:
            PULP_Nodes_Graph, class_out = self.generate_intermediate_activations(PULP_Nodes_Graph, 
                load_dir, 
                number_of_deployed_layers, 
                check_layer,
                weights_to_write)
        if check_layer == 100:
            act_compare = np.asarray([0, 0])
            act_size = [0, 0, 0]
        else:
            act_size = [PULP_Nodes_Graph[check_layer].output_dim[0], PULP_Nodes_Graph[check_layer].output_dim[1], PULP_Nodes_Graph[check_layer].ch_out]
        ## printf the network file. It calls all the layer functions
        template.print_template_network(
            weights_files_list,
            PULP_Nodes_Graph[:number_of_deployed_layers],
            'char',
            name=name_list,
            test=True,
            has_bias=True,
            verbose_level=verbose_level,
            performance_single_layer = performance_single_layer,
            check_layer=check_layer,
            act_compare=act_compare,
            act_size=act_size,
            class_out=class_out,
            l1_buffer=L1_dimension,
            master_stack = master_stack,
            slave_stack = slave_stack,
            l2_buffer_size = l2_buffer_size,
            fc_frequency = fc_frequency,
            cl_frequency = cl_frequency,
            MACs=MAC_total,
            platform=self.platform,
            sdk = sdk,
            backend = backend,
            dma_parallelization = dma_parallelization)
        # create the Makefile for the application
        template.print_template_Makefile(weights_files_list, self.platform, sdk, backend)
