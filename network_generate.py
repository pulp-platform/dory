#!/usr/bin/env python
# network_generate.py
# Alessio Burrello <alessio.burrello@unibo.it>
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

#####################CONFIG PARAMETERS #########################
# BNRelu_bits. Number of bits for lambda and k parameters in BNRelu. 32 or 64
# onnx file.

# Libraries
import argparse
import os.path
from argparse import RawTextHelpFormatter
import json
from importlib import import_module


def dory_to_c(graph, target, conf, confdir, verbose_level, perf_layer, optional, appdir, n_inputs):
    # Including and running the transformation from DORY IR to DORY HW IR
    onnx_manager = import_module(f'dory.Hardware_targets.{target}.HW_Parser')
    dory_to_dory_hw = onnx_manager.onnx_manager
    graph = dory_to_dory_hw(graph, conf, confdir, n_inputs).full_graph_parsing()

    # Deployment of the model on the target architecture
    onnx_manager = import_module(f'dory.Hardware_targets.{target}.C_Parser')
    dory_hw_to_c = onnx_manager.C_Parser
    dory_hw_to_c(graph, conf, confdir, verbose_level, perf_layer, optional, appdir, n_inputs).full_graph_parsing()


def network_generate(frontend, target, conf_file, verbose_level='Check_all+Perf_final', perf_layer='No', optional='auto',
                     appdir='./application', prefix="", n_trainable_layers=0):
    print(f"Using {frontend} as frontend. Targeting {target} platform. ")

    if len(prefix) > 0 and prefix[-1] != "_":
        prefix += "_"
    # Reading the json configuration file
    with open(conf_file) as f:
        conf = json.load(f)

    try:
        n_inputs = conf["n_inputs"]
    except KeyError:
        n_inputs = 1
    if n_inputs != 1:
        assert n_inputs > 1, "n_inputs must be >= 1!"

    # Reading the onnx file
    confdir = os.path.dirname(conf_file)
    onnx_file = os.path.join(confdir, conf["onnx_file"])
    print(f"Using {onnx_file} target input onnx.\n")

    # Including and running the transformation from Onnx to a DORY compatible graph
    onnx_manager = import_module(f'dory.Frontend_frameworks.{frontend}.Parser')
    onnx_to_dory = onnx_manager.onnx_manager
    graph = onnx_to_dory(onnx_file, conf, prefix, n_trainable_layers).full_graph_parsing()

    dory_to_c(graph, target, conf, confdir, verbose_level, perf_layer, optional, appdir, n_inputs)


if __name__ == '__main__':
    Frontends = ["NEMO", "Quantlab"]
    Hardware_targets = ["PULP.GAP8", "PULP.GAP8_L2", "PULP.PULP_gvsoc", "PULP.GAP9", "PULP.GAP9_NE16", "Occamy", "Diana.Diana_TVM", "Diana.Diana_SoC"]
    verbose_levels = ["None", "Perf_final", "Check_all+Perf_final", "Last+Perf_final"]
    optional_choices = ["auto", "8bit", "mixed-hw", "mixed-sw"]

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('frontend', type=str, choices=Frontends, help='Frontend from which the onnx is produced and from which the network has been trained')
    parser.add_argument('hardware_target', type=str, choices=Hardware_targets, help='Hardware platform for which the code is optimized')
    parser.add_argument('config_file', type=str, help='Path to the JSON file that specifies the ONNX file of the network and other information.')
    parser.add_argument('--verbose_level', choices=verbose_levels, default='Check_all+Perf_final',
                        help="None: No_printf.\n"
                             "Perf_final: only total performance\n"
                             "Check_all+Perf_final: all check + final performances \n"
                             "Last+Perf_final: all check + final performances \n"
                             "Extract the parameters from the onnx model")
    parser.add_argument('--perf_layer', action='store_true', help='Print the performance of each layer.')
    parser.add_argument('--optional', default='auto', choices=optional_choices,
                        help='auto (based on layer precision, 8bits or mixed-sw), 8bit, mixed-hw, mixed-sw')
    parser.add_argument('--app_dir', default='./application', help='Path to the generated application. Default: ./application')
    parser.add_argument('--prefix', default="", help='Prefix to prepend to network-specific generated functions', type=str)
    parser.add_argument('--n_trainable_layers', default=0, type=int, help='Number of layers that will be deployed externally.')

    args = parser.parse_args()

    network_generate(args.frontend, args.hardware_target, args.config_file, args.verbose_level, 'Yes' if args.perf_layer else 'No',
                     args.optional, args.app_dir, args.prefix, args.n_trainable_layers)
