     # should work even without -*-
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

import sys
sys.path.append('../')
from Model_deployment import Model_deployment 
import os
from collections import OrderedDict
from mako.template import Template
import numpy as np
import torch
import pandas as pd

class Model_deployment_MCU(Model_deployment):
    """
    Used to manage the PULP graph. By now, supported Convolutions, Pooling, Linear Layers and Relu.
    """

    def __init__(self, platform, chip):
        Model_deployment.__init__(self, platform, chip)


    def copy_backend(self, BitActivation, PULP_Nodes_Graph, number_of_deployed_layers, sdk, backend, dma_parallelization, optional):
        layer_mixed_list = []
        ####################################################################################
        ###### SECTION 1: BACKEND FILE SELECTING. SELECTING CORRECT KERNELS TO IMPORT ######
        ####################################################################################
        if optional == 'auto':
            optional = '8bits'
            for node in PULP_Nodes_Graph:
                if 'Conv' in node.get_parameter('name'):
                    ### NOT WORKING IF NO ANNOTATION IS PRESENT IN THE GRAPH: E.G. FOR NEMO
                    if node.get_parameter('out_activation_bits') < 8 or node.get_parameter('input_activation_bits') < 8 or node.get_parameter('weight_bits') < 8:
                        optional = 'mixed-sw'
                    ### Should be 3 in case of 1D convolution: each dimension is equal to 1
                    h_dimension = node.get_parameter('kernel_shape')[0] + node.get_parameter('input_dim')[0] + node.get_parameter('output_dim')[0]
                    if h_dimension == 3:
                        optional += '1DConv'
            ####### 1D CONV MIXED KERNELS NOT PRESENT ######
            if 'mixed-sw' in optional:
                optional = 'mixed-sw'
        else:
            pass
        if 'mixed-sw' in optional:
            for i, nodes_to_deploy in enumerate(PULP_Nodes_Graph[:number_of_deployed_layers]):
                BitIn = PULP_Nodes_Graph[i].input_activation_bits
                BitOut = PULP_Nodes_Graph[i].out_activation_bits
                if ('Pool' not in PULP_Nodes_Graph[i].name) and ('Add' not in PULP_Nodes_Graph[i].name):
                    BitW = PULP_Nodes_Graph[i].weight_bits
                if 'DW' in PULP_Nodes_Graph[i].name:
                    layer_mixed_list.append(f'pulp_nn_depthwise_u{BitIn}_u{BitOut}_i{BitW}.c')
                elif 'Conv' in PULP_Nodes_Graph[i].name:
                    layer_mixed_list.append(f'pulp_nn_conv_u{BitIn}_u{BitOut}_i{BitW}.c')
                if ('Conv' in PULP_Nodes_Graph[i].name or 'Gemm' in PULP_Nodes_Graph[i].name or 'MatMul' in PULP_Nodes_Graph[i].name) and BitOut!=32:
                    layer_mixed_list.append(f'pulp_nn_matmul_u{BitOut}_i{BitW}.c')
                if 'Gemm' in nodes_to_deploy.name or 'MatMul' in nodes_to_deploy.name:
                    if BitOut==32:
                        layer_mixed_list.append(f'pulp_nn_linear_u{BitIn}_i{BitOut}_i{BitW}.c')
                    else:
                        layer_mixed_list.append(f'pulp_nn_linear_u{BitIn}_u{BitOut}_i{BitW}.c')
            layer_mixed_list.append('pulp_nn_add_u8_u8.c')
            layer_mixed_list.append('pulp_nn_avgpool_u8.c')
            layer_mixed_list.append('pulp_nn_maxpool_u8.c')
            layer_mixed_list.append('pulp_nn_avgpool_u4.c')
            layer_mixed_list.append('pulp_nn_maxpool_u4.c')
            layer_mixed_list.append('pulp_nn_avgpool_u2.c')
            layer_mixed_list.append('pulp_nn_maxpool_u2.c')
        if 'mixed-hw' in optional:
            for i, nodes_to_deploy in enumerate(PULP_Nodes_Graph[:number_of_deployed_layers]):
                BitIn = PULP_Nodes_Graph[i].input_activation_bits
                BitOut = PULP_Nodes_Graph[i].out_activation_bits
                BitW = PULP_Nodes_Graph[i].weight_bits
                if 'DW' in PULP_Nodes_Graph[i].name:
                    layer_mixed_list.append(f'xpulp_nn_depthwise_u{BitIn}_u{BitOut}_i{BitW}.c')
                elif 'Conv' in PULP_Nodes_Graph[i].name:
                    layer_mixed_list.append(f'xpulp_nn_conv_u{BitIn}_u{BitOut}_i{BitW}.c')
                if ('Conv' in PULP_Nodes_Graph[i].name or 'Gemm' in PULP_Nodes_Graph[i].name or 'MatMul' in PULP_Nodes_Graph[i].name) and BitOut!=32:
                    layer_mixed_list.append(f'xpulp_nn_matmul_u{BitIn}_u{BitOut}_i{BitW}.c')
                if 'Gemm' in nodes_to_deploy.name or 'MatMul' in nodes_to_deploy.name:
                    if BitOut==32:
                        layer_mixed_list.append(f'pulp_nn_linear_u{BitIn}_i{BitOut}_i{BitW}.c')
                    else:
                        layer_mixed_list.append(f'pulp_nn_linear_u{BitIn}_u{BitOut}_i{BitW}.c')
            layer_mixed_list.append('pulp_nn_add_u8_u8.c')
            layer_mixed_list.append('pulp_nn_avgpool_u8.c')
            layer_mixed_list.append('pulp_nn_maxpool_u8.c')
            layer_mixed_list.append('pulp_nn_avgpool_u4.c')
            layer_mixed_list.append('pulp_nn_maxpool_u4.c')
            layer_mixed_list.append('pulp_nn_avgpool_u2.c')
            layer_mixed_list.append('pulp_nn_maxpool_u2.c')
        version = str(BitActivation) + 'bit'
        self.copy_files(optional, layer_mixed_list, version, sdk, backend, dma_parallelization)

    def copy_files(self, optional, layer_mixed_list,version, sdk, backend, dma_parallelization):
        ## copy backend and necessary files in the application folder
        os.system('rm -rf application')
        os.system('mkdir application')
        os.system('mkdir application/DORY_network')
        os.system('mkdir application/DORY_network/inc')
        os.system('mkdir application/DORY_network/src')
        tk = OrderedDict([])
        tk['sdk'] = sdk
        root = '/'.join(os.getcwd().split('/')[:-1])
        tmpl = Template(filename=root + f"/Templates/{backend}/dory.h")
        s = tmpl.render(**tk)
        save_string = './application/DORY_network/inc/dory.h'
        with open(save_string, "w") as f:
            f.write(s)
        os.system(f'cp ../Templates/{backend}/mem_controller.c  ./application/DORY_network/src/')
        os.system(f'cp ../Templates/{backend}/mem_controller.h  ./application/DORY_network/inc/')
        tk = OrderedDict([])
        tk['sdk'] = sdk
        tmpl = Template(filename=root+f"/Templates/{backend}/mchan_test.h")
        s = tmpl.render(**tk)
        save_string = './application/DORY_network/inc/mchan_test.h'
        with open(save_string, "w") as f:
            f.write(s)
        tk = OrderedDict([])
        tk['chip'] = self.chip
        tk['dma_parallelization'] = dma_parallelization
        tmpl = Template(filename=root+f"/Templates/{backend}/dory.c")
        s = tmpl.render(**tk)
        save_string = './application/DORY_network/src/dory.c'
        with open(save_string, "w") as f:
            f.write(s)
        os.system(f'cp ../Templates/{backend}/test_template.c ./application/DORY_network/src/')
        os.system(f'cp ../Templates/{backend}/network.h ./application/DORY_network/inc/')
        if "1DConv" in optional:
            os.system('cp ../Backend_Kernels/pulp-nn-1d/' + version +'/include/*  ./application/DORY_network/inc/')
            os.system('cp ../Backend_Kernels/pulp-nn-1d/' + version +'/src/* ./application/DORY_network/src/')
        elif "8bit" in optional:
            os.system('cp ../Backend_Kernels/pulp-nn/' + version +'/include/*  ./application/DORY_network/inc/')
            os.system('cp ../Backend_Kernels/pulp-nn/' + version +'/src/* ./application/DORY_network/src/')
        elif "mixed-sw" in optional:
            os.system('cp ../Backend_Kernels/pulp-nn-mixed/XpulpV2/' + version +'/include/*  ./application/DORY_network/inc/')
            for layer in layer_mixed_list:
                if layer.split('_')[2] == 'conv':
                    os.system('cp ../Backend_Kernels/pulp-nn-mixed/XpulpV2/' + version +'/src/Convolution/' + layer + ' ./application/DORY_network/src/')
                elif layer.split('_')[2] == 'depthwise':
                    os.system('cp ../Backend_Kernels/pulp-nn-mixed/XpulpV2/' + version +'/src/Depthwise/' + layer + ' ./application/DORY_network/src/')
                elif layer.split('_')[2] == 'matmul':
                    os.system('cp ../Backend_Kernels/pulp-nn-mixed/XpulpV2/' + version +'/src/MatrixMultiplication/' + layer + ' ./application/DORY_network/src/')
                elif layer.split('_')[2] == 'linear':
                    if layer.split('_')[4] == 'i32':
                        os.system('cp ../Backend_Kernels/pulp-nn-mixed/XpulpV2/' + version +'/src/LinearNoQuant/' + layer + ' ./application/DORY_network/src/')
                    else:
                        os.system('cp ../Backend_Kernels/pulp-nn-mixed/XpulpV2/' + version +'/src/LinearQuant/' + layer + ' ./application/DORY_network/src/')
                elif 'avgpool' in layer.split('_')[2]:
                    os.system('cp ../Backend_Kernels/pulp-nn-mixed/XpulpV2/' + version +'/src/Pooling/AvgPool/' + layer + ' ./application/DORY_network/src/')
                elif 'maxpool' in layer.split('_')[2]:
                    os.system('cp ../Backend_Kernels/pulp-nn-mixed/XpulpV2/' + version +'/src/Pooling/MaxPool/' + layer + ' ./application/DORY_network/src/')
                elif layer.split('_')[2] == 'add':
                    os.system('cp ../Backend_Kernels/pulp-nn-mixed/XpulpV2/' + version +'/src/Add/' + layer + ' ./application/DORY_network/src/')
        elif "mixed-hw" in optional:
            os.system('cp ../Backend_Kernels/pulp-nn-mixed/XpulpNN/' + version +'/include/*  ./application/DORY_network/inc/')
            for layer in layer_mixed_list:
                if layer.split('_')[2] == 'conv':
                    os.system('cp ../Backend_Kernels/pulp-nn-mixed/XpulpNN/' + version +'/src/Convolution/' + layer + ' ./application/DORY_network/src/')
                elif layer.split('_')[2] == 'depthwise':
                    os.system('cp ../Backend_Kernels/pulp-nn-mixed/XpulpNN/' + version +'/src/Depthwise/' + layer + ' ./application/DORY_network/src/')
                elif layer.split('_')[2] == 'matmul':
                    os.system('cp ../Backend_Kernels/pulp-nn-mixed/XpulpNN/' + version +'/src/MatrixMultiplication/' + layer + ' ./application/DORY_network/src/')
                elif layer.split('_')[2] == 'linear':
                    if layer.split('_')[4] == 'i32':
                        os.system('cp ../Backend_Kernels/pulp-nn-mixed/XpulpNN/' + version +'/src/LinearNoQuant/' + layer + ' ./application/DORY_network/src/')
                    else:
                        os.system('cp ../Backend_Kernels/pulp-nn-mixed/XpulpNN/' + version +'/src/LinearQuant/' + layer + ' ./application/DORY_network/src/')
                elif 'avgpool' in layer.split('_')[2]:
                    os.system('cp ../Backend_Kernels/pulp-nn-mixed/XpulpNN/' + version +'/src/Pooling/AvgPool/' + layer + ' ./application/DORY_network/src/')
                elif 'maxpool' in layer.split('_')[2]:
                    os.system('cp ../Backend_Kernels/pulp-nn-mixed/XpulpNN/' + version +'/src/Pooling/MaxPool/' + layer + ' ./application/DORY_network/src/')
                elif layer.split('_')[2] == 'add':
                    os.system('cp ../Backend_Kernels/pulp-nn-mixed/XpulpNN/' + version +'/src/Add/' + layer + ' ./application/DORY_network/src/')

    def create_weights_files(self, PULP_Nodes_Graph, number_of_deployed_layers, BitActivation, load_dir):
        ####################################################################################
        ###### SECTION 2: WEIGHTS FILES CREATION. CREATING .HEX FILES FOR EACH LAYER  ######
        ####################################################################################
        file_list_w = []
        # Fetching weights,biases, k, and lambda for each node_iterating
        # 32 bits and 64 bits for Bn and Relu weights are used
        weights_to_write = []
        for i, nodes_to_deploy in enumerate(PULP_Nodes_Graph[:number_of_deployed_layers]):
            if 'weights' in nodes_to_deploy.__dict__:
                if PULP_Nodes_Graph[i].weight_bits < 8 and 'DW' in nodes_to_deploy.name:
                    nodes_to_deploy.weights = nodes_to_deploy.weights.reshape(int(nodes_to_deploy.weights.shape[0]/2),2,nodes_to_deploy.weights.shape[1],nodes_to_deploy.weights.shape[2],nodes_to_deploy.weights.shape[3]).transpose(0,2,3,1,4).flatten().tolist()
                else:
                    nodes_to_deploy.weights = nodes_to_deploy.weights.flatten().tolist()
                for i_w, _ in enumerate(nodes_to_deploy.weights):
                    nodes_to_deploy.weights[i_w] = np.uint8(nodes_to_deploy.weights[i_w])
                if PULP_Nodes_Graph[i].weight_bits == 4:
                    temp = []
                    z = 0
                    for i_x, _ in enumerate(nodes_to_deploy.weights):
                        if (z % 2) == 0:
                            temp.append(nodes_to_deploy.weights[i_x]& 0x0F)
                        else:
                            temp[-1] += (nodes_to_deploy.weights[i_x]& 0x0F) << 4
                        z += 1
                    nodes_to_deploy.weights = temp
                elif PULP_Nodes_Graph[i].weight_bits == 2:
                    temp = []
                    z = 0
                    for i_x, _ in enumerate(nodes_to_deploy.weights):
                        if (z % 4) == 0:
                            temp.append(nodes_to_deploy.weights[i_x]& 0x03)
                        else:
                            temp[-1] += (nodes_to_deploy.weights[i_x]& 0x03) << 2 * (z % 4)
                        z += 1
                    nodes_to_deploy.weights = temp
                weights = nodes_to_deploy.weights
            if 'bias' in nodes_to_deploy.__dict__:
                nodes_to_deploy.bias = nodes_to_deploy.bias.flatten().tolist()
                for i_w, _ in enumerate(nodes_to_deploy.bias):
                    nodes_to_deploy.bias[i_w] = np.uint8(nodes_to_deploy.bias[i_w])
                weights = np.concatenate((weights, nodes_to_deploy.bias))
            if 'k' in nodes_to_deploy.__dict__:
                k_byte = []
                for i_k, _ in enumerate(nodes_to_deploy.k.flatten()):
                    if BitActivation == 64:
                        val = np.int64(nodes_to_deploy.k.flatten()[i_k])
                    else:
                        val = np.int32(nodes_to_deploy.k.flatten()[i_k])
                    if BitActivation == 32:
                        k_byte.append(np.uint8(val         & 0x000000FF))
                        k_byte.append(np.uint8((val >> 8)  & 0x000000FF))
                        k_byte.append(np.uint8((val >> 16) & 0x000000FF))
                        k_byte.append(np.uint8((val >> 24) & 0x000000FF))
                    if BitActivation == 64:
                        k_byte.append(np.uint8(val         & 0x00000000000000FF))
                        k_byte.append(np.uint8((val >> 8)  & 0x00000000000000FF))
                        k_byte.append(np.uint8((val >> 16) & 0x00000000000000FF))
                        k_byte.append(np.uint8((val >> 24) & 0x00000000000000FF))
                        k_byte.append(np.uint8((val >> 32) & 0x00000000000000FF))
                        k_byte.append(np.uint8((val >> 40) & 0x00000000000000FF))
                        k_byte.append(np.uint8((val >> 48) & 0x00000000000000FF))
                        k_byte.append(np.uint8((val >> 56) & 0x00000000000000FF))
                nodes_to_deploy.k = k_byte
                weights = np.concatenate((weights, nodes_to_deploy.k))
            if 'lambda' in nodes_to_deploy.__dict__:
                lambd = np.float64(nodes_to_deploy.get_parameter('lambda').flatten())
                try:
                    lambd.shape[0]
                except:
                    lambd = np.asarray([np.float64(nodes_to_deploy.get_parameter('lambda').flatten())])
                lambd_byte = []
                for i_l, _ in enumerate(nodes_to_deploy.get_parameter('lambda').flatten()):
                    if BitActivation == 64:
                        val = np.int64(lambd[i_l])
                    else:
                        val = np.int32(lambd[i_l])
                    if BitActivation == 32:
                        lambd_byte.append(np.uint8(val &         0x000000FF))
                        lambd_byte.append(np.uint8((val >> 8) &  0x000000FF))
                        lambd_byte.append(np.uint8((val >> 16) & 0x000000FF))
                        lambd_byte.append(np.uint8((val >> 24) & 0x000000FF))
                    if BitActivation == 64:
                        lambd_byte.append(np.uint8(val &         0x00000000000000FF))
                        lambd_byte.append(np.uint8((val >> 8) &  0x00000000000000FF))
                        lambd_byte.append(np.uint8((val >> 16) & 0x00000000000000FF))
                        lambd_byte.append(np.uint8((val >> 24) & 0x00000000000000FF))
                        lambd_byte.append(np.uint8((val >> 32) & 0x00000000000000FF))
                        lambd_byte.append(np.uint8((val >> 40) & 0x00000000000000FF))
                        lambd_byte.append(np.uint8((val >> 48) & 0x00000000000000FF))
                        lambd_byte.append(np.uint8((val >> 56) & 0x00000000000000FF))
                nodes_to_deploy.add_parameter('lambda', lambd_byte)
                weights = np.concatenate((weights, nodes_to_deploy.get_parameter('lambda')))
            if 'weights' in nodes_to_deploy.__dict__:
                while len(weights) % 4 != 0:
                    weights = np.concatenate((weights, np.asarray([0])))
                weights = np.asarray(weights)
                weights_to_write.append(weights)
                string_layer = nodes_to_deploy.name + str(i) + "_weights.hex"
                file_list_w.append(string_layer)
                save_s = './application/DORY_network/' + string_layer
                with open(save_s, 'wb') as f:
                    for l in weights.astype('uint8').flatten():
                        f.write(bytes((l,)))
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
        string_layer = "inputs.hex"
        save_s = './application/DORY_network/' + string_layer
        with open(save_s, 'wb') as f:
            for i in x_in.astype('uint8').flatten():
                f.write(bytes((i,)))

        return PULP_Nodes_Graph, file_list_w, weights_to_write