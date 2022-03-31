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
import sys

def print_test_vector(x, data_type):
    if data_type == 'uint8_t':
        try:
            np.set_printoptions(
                threshold=sys.maxsize,
                formatter={'float': lambda x: (np.float32(x)), }
            )
        except TypeError:
            np.set_printoptions(threshold=sys.maxsize)
    elif data_type == 'int8_t':
        x = x.astype(np.int32)
        try:
            np.set_printoptions(
                threshold=sys.maxsize,
                formatter={'float': lambda x: (np.float32(x)), }
            )
        except TypeError:
            np.set_printoptions(threshold=sys.maxsize)
    s = repr(x.flatten()).replace("array([", "").replace("]", "").replace("[", "").replace(")", "").replace(",\n      dtype=int8)", "").replace(",\n      dtype=int32)", "").replace(", dtype=uint8", "").replace(",\n      dtype=uint8)", "").replace(",\n      dtype=uint8", "").replace(",\n      dtype=int8", "").replace(",\n      dtype=int32", "").replace(", dtype=int8", "").replace(", dtype=int32", "").replace(", dtype=int8)", "").replace(", dtype=int8)", "").replace(", dtype=uint8)", "")
     
    return s

class Model_deployment_Occamy(Model_deployment):
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
        optional = '8bits'
        for node in PULP_Nodes_Graph:
            if 'Conv' in node.get_parameter('name'):
                if node.get_parameter('out_activation_bits') < 8 or node.get_parameter('input_activation_bits') < 8 or node.get_parameter('weight_bits') < 8:
                    optional = 'mixed-sw'
                ### Should be 3 in case of 1D convolution: each dimension is equal to 1
                h_dimension = node.get_parameter('kernel_shape')[0] + node.get_parameter('input_dim')[0] + node.get_parameter('output_dim')[0]
                if h_dimension == 3:
                    optional += '1DConv'
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
        root = '/'.join(os.getcwd().split('/')[:-1])
        tmpl = Template(filename=root + f"/Templates/{backend}/dory.h")
        s = tmpl.render(**tk)
        save_string = './application/DORY_network/inc/dory.h'
        with open(save_string, "w") as f:
            f.write(s)
        os.system(f'cp ../Templates/{backend}/mem_controller.c  ./application/DORY_network/src/')
        os.system(f'cp ../Templates/{backend}/mem_controller.h  ./application/DORY_network/inc/')
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
        os.system('cp ../Backend_Kernels/Occamy/include/*  ./application/DORY_network/inc/')
        os.system('cp ../Backend_Kernels/Occamy/src/* ./application/DORY_network/src/')

    def create_weights_files(self, PULP_Nodes_Graph, number_of_deployed_layers, BitActivation, load_dir):
        file_list_w = []
        # Fetching weights,biases, k, and lambda for each node_iterating
        # 32 bits and 64 bits for Bn and Relu weights are used
        weights_to_write = []
        weights_to_write_h = []
        for i, nodes_to_deploy in enumerate(PULP_Nodes_Graph[:number_of_deployed_layers]):
            if 'weights' in nodes_to_deploy.__dict__:
                if PULP_Nodes_Graph[i].weight_bits < 8 and 'DW' in nodes_to_deploy.name:
                    nodes_to_deploy.weights = nodes_to_deploy.weights.reshape(int(nodes_to_deploy.weights.shape[0]/2),2,nodes_to_deploy.weights.shape[1],nodes_to_deploy.weights.shape[2],nodes_to_deploy.weights.shape[3]).transpose(0,2,3,1,4).flatten().tolist()
                else:
                    kernel_shape = nodes_to_deploy.get_parameter('kernel_shape')
                    if i == 0 and kernel_shape[0]*kernel_shape[1] % 2 != 0:
                        nodes_to_deploy.weights = np.concatenate((nodes_to_deploy.weights, np.zeros((nodes_to_deploy.ch_out,nodes_to_deploy.kernel_shape[0],1,nodes_to_deploy.ch_in))),axis = 2)
                        PULP_Nodes_Graph[i].kernel_shape[1] = PULP_Nodes_Graph[i].kernel_shape[1]+1
                        PULP_Nodes_Graph[i].pads[3] = PULP_Nodes_Graph[i].pads[3]+1
                        PULP_Nodes_Graph[i].weights = nodes_to_deploy.weights
                    nodes_to_deploy.weights = nodes_to_deploy.weights.flatten().tolist()
                for i_w, _ in enumerate(nodes_to_deploy.weights):
                    nodes_to_deploy.weights[i_w] = np.int8(nodes_to_deploy.weights[i_w])
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
                    k_byte.append(val)
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
                    lambd_byte.append(val)
                nodes_to_deploy.add_parameter('lambda', lambd_byte)
                weights = np.concatenate((weights, nodes_to_deploy.get_parameter('lambda')))
            if 'weights' in nodes_to_deploy.__dict__:
                while len(weights) % 4 != 0:
                    weights = np.concatenate((weights, np.asarray([0])))
                weights = np.asarray(weights)
                weights_to_write.append(weights)
                weights_to_write_h.append(print_test_vector(weights, 'int8_t'))
                string_layer = nodes_to_deploy.name + str(i) + "_weights"
                file_list_w.append(string_layer)
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
        tk = OrderedDict([])
        tk['weights_layers_names'] = file_list_w
        tk['weight_layers'] = weights_to_write_h
        tk['weight_layers_for_dim'] = weights_to_write
        tk['input'] = print_test_vector(x_in, 'uint8_t')
        tk['input_len'] = len(x_in)
        root = '/'.join(os.getcwd().split('/')[:-1])
        tmpl = Template(filename=root + f"/Templates/Occamy/data.h")
        s = tmpl.render(**tk)
        save_string = './application/DORY_network/inc/data.h'
        with open(save_string, "w") as f:
            f.write(s)
        return PULP_Nodes_Graph, file_list_w, weights_to_write