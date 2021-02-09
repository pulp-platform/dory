import onnx
from onnx import numpy_helper
from onnx import helper, shape_inference
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
from collections import OrderedDict
import logging

class node_element_1D(nn.Module):
    # A node allocated in the PULP_Graph
    def __init__(self):
        self.name = 'empty'
        self.filter_size_h = 1
        self.filter_size_w = 1
        self.input_channels = 0
        self.output_channels = 0
        self.padding_top = 0
        self.padding_bottom = 0
        self.padding_left   = 0
        self.padding_right  = 0
        self.stride = 1
        self.groups = 1
        self.weights = 'empty'
        self.k = 'empty'
        self.lambd = 'empty'
        self.outmul = 'empty'
        self.inmul1 = 'empty'
        self.inmul2 = 'empty'
        self.outshift = 'empty'
        self.bias = 'empty'
        self.input_index = 0
        self.input_index_add = 0
        self.output_index = 0
        self.input_h = 0
        self.input_w = 0
        self.output_h = 0
        self.output_w = 0
        self.L3_allocation = 0
        self.input_activation_dimensions = 0
        self.output_activation_dimensions = 0
        self.check_sum_in = 0
        self.check_sum_out = 0
        self.check_sum_w = 0
        self.l1_dimensions = 0
        self.branch_out = 0
        self.dilation = 1
        self.branch_in = 0
        self.weights_dimension = 0
        self.weights_dimension_L3 = 0
        self.MACs = 0


class ONNX_management_1D():
    # Used to manage the ONNX files. By now, supported Convolutions (PW and DW), Pooling, Fully Connected and Relu.

    def __init__(self, platform, chip, network="model_to_convert.onnx"):
        self.network = network
        self.platform = platform
        self.chip = chip

    def create_node_add(self, new_node, first_node, node_iterating, model, PULP_Nodes_Graph):
        # Allocation of an Addition (for residuals) layer
        new_node.input_index = [input_i for input_i in node_iterating.input if 'weight' not in input_i][0]
        new_node.input_index_add = [input_i for input_i in node_iterating.input if 'weight' not in input_i][1]
        new_node.output_index = node_iterating.output[0]
        for nodes in PULP_Nodes_Graph:
            if nodes.output_index == new_node.input_index:
                new_node.input_h = nodes.output_h
                new_node.input_w = nodes.output_w
                new_node.input_channels = nodes.output_channels
                new_node.output_channels = nodes.output_channels
                new_node.output_h = nodes.output_h
                new_node.output_w = nodes.output_w
        new_node.name = node_iterating.op_type

        return new_node



    def create_node(self, new_node, first_node, node_iterating, model, PULP_Nodes_Graph):
        # Allocation of a Node, Convolution, Pooling or Linear
        new_node.input_index = [input_i for input_i in node_iterating.input if 'weight' not in input_i][0]
        new_node.output_index = node_iterating.output[0]
        new_node.padding_left   = 0
        new_node.padding_right  = 0
        # scan pad layer
        for node in model.graph.node:
            if node.output[0] in new_node.input_index and node.op_type == 'Pad':
                new_node.input_index = node.input[0]
                if(sum(node.attribute[1].ints)>0):
                    new_node.padding_left+=node.attribute[1].ints[2]
                    new_node.padding_right+=node.attribute[1].ints[5]
        if first_node == 1:
            if(len(model.graph.input[0].type.tensor_type.shape.dim)==3):
                new_node.input_h = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
                new_node.input_w = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
            else:
                new_node.input_h = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
                new_node.input_w = model.graph.input[0].type.tensor_type.shape.dim[3].dim_value
        else:
            if 'Gemm' not in node_iterating.op_type and 'MatMul' not in node_iterating.op_type:
                for nodes in PULP_Nodes_Graph:
                    if nodes.output_index == new_node.input_index:
                        new_node.input_h = nodes.output_h
                        new_node.input_w = nodes.output_w
            else:
                new_node.input_h = 1
                new_node.input_w = 1
        new_node.name = node_iterating.op_type
        try:
            if 'MatMul' == node_iterating.op_type:
            	weight_name = [input_i for input_i in node_iterating.input][1]
            else:
            	weight_name = [input_i for input_i in node_iterating.input if 'weight' in input_i][0]
        except:
            weight_name = 'NotFound'
        try:
            bias_name = [input_i for input_i in node_iterating.input if 'bias' in input_i][0]
        except:
            bias_name = 'NotFound'
        if 'Conv' in node_iterating.op_type:
            for weight in model.graph.initializer:
                if weight.name == weight_name:
                    new_node.weights = np.transpose(numpy_helper.to_array(weight), (0, 2, 1))
                    new_node.input_channels = weight.dims[1]
                    new_node.output_channels = weight.dims[0]
        elif 'Gemm' in node_iterating.op_type or 'MatMul' in node_iterating.op_type:
            for weight in model.graph.initializer:
                if weight.name == weight_name:
                    temp = numpy_helper.to_array(weight)
                    if 'MatMul' in node_iterating.op_type:
                    	temp = temp.reshape(temp.shape[1], PULP_Nodes_Graph[-1].output_channels, PULP_Nodes_Graph[-1].output_h, PULP_Nodes_Graph[-1].output_w)
                    else:
                        temp = temp.reshape(temp.shape[0], PULP_Nodes_Graph[-1].output_channels, PULP_Nodes_Graph[-1].output_h, PULP_Nodes_Graph[-1].output_w)
                    temp = np.transpose(temp, (0, 2, 3, 1))
                    temp = temp.flatten()
                    new_node.weights = temp
                    if 'MatMul' in node_iterating.op_type:
                        new_node.input_channels = weight.dims[0]
                        new_node.output_channels = weight.dims[1]
                    else:
                        new_node.input_channels = weight.dims[1]
                        new_node.output_channels = weight.dims[0]
                if weight.name == bias_name:
                    new_node.bias = numpy_helper.to_array(weight)
        elif 'Pool' in node_iterating.op_type:
            new_node.input_channels = PULP_Nodes_Graph[-1].output_channels
            new_node.output_channels = PULP_Nodes_Graph[-1].output_channels
        if 'Gemm' not in node_iterating.op_type and 'MatMul' not in node_iterating.op_type:
            for field in node_iterating.attribute:
                if field.name == 'kernel_shape':
                    new_node.filter_size_h = 1
                    new_node.filter_size_w = field.ints[0]
                if field.name == 'pads':
                    new_node.padding_bottom += field.ints[0]
                    new_node.padding_right  += field.ints[1]
                if field.name == 'strides':
                    new_node.stride = field.ints[0]
                if field.name == 'group':
                    new_node.groups = field.i
            if new_node.groups > 1:
                new_node.name = new_node.name + 'DW'
        elif 'Gemm' in node_iterating.op_type or 'MatMul' in node_iterating.op_type:
            new_node.filter_size_h = 1
            new_node.filter_size_w = 1
            new_node.padding_top    = 0
            new_node.padding_left   = 0
            new_node.padding_bottom = 0
            new_node.padding_right  = 0
            new_node.stride = 1
        if 'Gemm' not in node_iterating.op_type and 'MatMul' not in node_iterating.op_type:
            new_node.output_h = 1
            #NEED TO CHANGE WITH DILATION....
            new_node.output_w = int(np.ceil((new_node.input_w - (new_node.filter_size_w - 1) + new_node.padding_left + new_node.padding_right) / new_node.stride))
        else:
            new_node.output_h = 1
            new_node.output_w = 1
        return new_node

    def search_constant(self, index, model):
        ## searching for the parameters of BN abd Relu
        constant = 'empty'
        for node_iterating in (model.graph.initializer):
            if node_iterating.name == index:
                constant = numpy_helper.to_array(node_iterating)
        for node_iterating in (model.graph.node):
            if node_iterating.op_type == 'Constant' and node_iterating.output[0] == index:
                constant = numpy_helper.to_array(node_iterating.attribute[0].t)
        return constant

    def update_node(self, PULP_node, out_index, const, op_type):
        # Add BN and Relu to the nodes.
        PULP_node.output_index = out_index
        if str(const) != 'empty':
            if op_type == 'Add':
                PULP_node.lambd = const
                PULP_node.name = PULP_node.name + 'BN'
            elif op_type == 'Div':
                try:
                    const[0]
                    PULP_node.outshift = round(np.log2(const[0]))
                except:
                    PULP_node.outshift = round(np.log2(const))
                PULP_node.name = PULP_node.name + 'Relu'
            elif op_type == 'Mul':
                try:
                    const.shape[0]
                    ### TO FIX FOR SINGLE CHANNEL OUTPUT
                    if len(const.flatten())!=1:
                        PULP_node.k = const
                    else:
                        if str(PULP_node.outmul) == 'empty':
                            PULP_node.outmul = const[0]
                        else:
                            PULP_node.outmul = const[0] * PULP_node.outmul                        
                except:
                    if str(PULP_node.outmul) == 'empty':
                        PULP_node.outmul = const
                    else:
                        PULP_node.outmul = const * PULP_node.outmul
        return PULP_node

    def print_PULP_graph(self, PULP_Nodes_Graph):
        # Logging function to report exported graph of PULP
        print("Creating annotated graph in Network_annotated_graph.log")
        logging.basicConfig(filename='logs/Network_annotated_graph.log',
                            format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)
        for nodes in PULP_Nodes_Graph:
            logging.debug(f'New node_iterating: {nodes.name}')
            if 'Conv' in nodes.name or 'Gemm' in nodes.name or 'MatMul' in nodes.name:
                logging.debug(f'Filter Dimension i_ch,fs1,fs2,o_ch: [{nodes.input_channels},{nodes.filter_size_h},{nodes.filter_size_h},{nodes.output_channels}]')
            logging.debug(f'Stride: {nodes.stride}')
            logging.debug(f'Padding: {nodes.padding_top}, {nodes.padding_left}, {nodes.padding_bottom}, {nodes.padding_right}')
            logging.debug(f'Groups {nodes.groups}')
            logging.debug(f'MACs {nodes.MACs}')
            logging.debug(f'In-Out dimensions: [{nodes.input_h},{nodes.input_w}], [{nodes.output_h},{nodes.output_w}]')
            if str(nodes.weights) != 'empty':
                logging.debug(f'Weigths: present ')
            else:
                logging.debug(f'Weights: empty')
            if str(nodes.k) != 'empty':
                logging.debug(f'k: present ')
            else:
                logging.debug(f'k: empty')
            if str(nodes.lambd) != 'empty':
                logging.debug(f'lambd: present ')
            else:
                logging.debug(f'lambd: empty')
            if str(nodes.outmul) != 'empty':
                logging.debug(f'outmul: present ')
            else:
                logging.debug(f'outmul: empty')
            if 'Add' in nodes.name:
                if str(nodes.inmul1) != 'empty':
                    logging.debug(f'inmul1: present ')
                else:
                    logging.debug(f'inmul1: empty')
                if str(nodes.inmul2) != 'empty':
                    logging.debug(f'inmul2: present ')
                else:
                    logging.debug(f'inmul2: empty')
            if str(nodes.outshift) != 'empty':
                logging.debug(f'outshift: present ')
            else:
                logging.debug(f'outshift: empty')
            logging.debug(f'Input branch: {nodes.branch_in}')
            logging.debug(f'Output branch: {nodes.branch_out}')
            logging.debug(f'Input: {nodes.input_index}')
            if 'Add' in nodes.name:
                logging.debug(f'     : {nodes.input_index_add}')
            logging.debug(f'Output: {nodes.output_index}')
            logging.debug(f' ')

    def parameters_from_onnx(self, maxL):
        # Load all parameters from the onnx model.
        model = onnx.load(self.network)
        PULP_Nodes_Graph = []
        first_node = 1
        for node_iterating in (model.graph.node[:(maxL * 15)]):
            print(node_iterating.op_type)
            node_to_scan = 1
            # Adding a new Conv, Pool or Linear layer
            if 'Pool' in node_iterating.op_type or 'Conv' in node_iterating.op_type or 'Gemm' in node_iterating.op_type or 'MatMul' in node_iterating.op_type:
                new_node = node_element_1D()
                new_node = self.create_node(new_node, first_node, node_iterating, model, PULP_Nodes_Graph)
                first_node = 0
                PULP_Nodes_Graph.append(new_node)
                node_to_scan = 0
            # Adding an addition layer
            elif 'Add' in node_iterating.op_type:
                new_node = node_element_1D()
                not_real_add = 0
                inputs = [input_i for input_i in node_iterating.input if 'weight' not in input_i]
                for inp in inputs:
                    if 'lamda' in inp:
                        not_real_add = 1
                for node_const in (model.graph.node):
                    for inp in inputs:
                        if inp == node_const.output[0] and 'Const' in node_const.op_type:
                            not_real_add = 1
                if not_real_add == 0:
                    node_to_scan = 0
                    new_node = self.create_node_add(new_node, first_node, node_iterating, model, PULP_Nodes_Graph)
                    PULP_Nodes_Graph.append(new_node)
            # updating nodes using other onnx nodes
            if node_to_scan == 1:
                inputs = [input_i for input_i in node_iterating.input if 'weight' not in input_i]
                for PULP_node in PULP_Nodes_Graph:
                    for inp in inputs:
                        if inp == PULP_node.output_index:
                            # insert BN and/or Relu. Note that you need Mul-Add-Mul-Div or only 
                            if 'Mul' == node_iterating.op_type or 'Div' in node_iterating.op_type or 'Add' in node_iterating.op_type:
                                index = [input_search for input_search in inputs if input_search != inp][0]
                                const = self.search_constant(index, model)
                                PULP_node = self.update_node(PULP_node, node_iterating.output[0], const, node_iterating.op_type)
                                break
                            # only update output index with floor, clip and this kind of nodes
                            elif 'Pad' not in node_iterating.op_type:
                                PULP_node.output_index = node_iterating.output[0]
                                break
        # updating branch in/out connections
        for i, nodes in enumerate(PULP_Nodes_Graph):
            counter = 0
            for nodes_scan in PULP_Nodes_Graph:
                if nodes.output_index == nodes_scan.input_index:
                    counter += 1
                if 'Add' in nodes_scan.name:
                    if nodes.output_index == nodes_scan.input_index_add:
                        counter += 1
            if counter > 1:
                PULP_Nodes_Graph[i].branch_out = 1
            if 'Add' in nodes.name:
                PULP_Nodes_Graph[i].branch_in = 1
        os.system('rm -rf logs/*log')
        # computing MACs per layer
        for i, nodes in enumerate(PULP_Nodes_Graph):
            PULP_Nodes_Graph[i].MACs = nodes.filter_size_h * nodes.filter_size_w * \
                nodes.output_channels * nodes.input_channels * nodes.output_h * nodes.output_w
        # printing graph
        self.print_PULP_graph(PULP_Nodes_Graph)
        return PULP_Nodes_Graph

    