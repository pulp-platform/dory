     # should work even without -*-
# -*- coding: utf-8 -*-
#!/bin/bash
# utils.py
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

import os
import onnx
import json
import numpy as np
from google.protobuf.json_format import MessageToJson
from google.protobuf.json_format import Parse


def loadtxt(filepath, dtype):
    return np.loadtxt(filepath, delimiter=',', dtype=dtype, usecols=[0])


class Printer:
    def __init__(self, folder):
        self.folder = folder
        self.jsondir = os.path.join(folder, 'json_files')
        self.onnxdir = os.path.join(folder, 'onnx_files')

        os.system(f'rm -rf {self.folder}')
        os.system(f'mkdir -p {self.jsondir}')
        os.system(f'mkdir -p {self.onnxdir}')

    def __info(self, file, filedir):
        print(f'Creating {file} in {filedir}')

    def print_onnx(self, filename, graph):
        filename += '.onnx'
        onnx.save_model(graph, os.path.join(self.onnxdir, filename))
        self.__info(filename, self.onnxdir)

    def print_json(self, filename, graph):
        filename += '.json'
        s = MessageToJson(graph)
        onnx_json = json.loads(s)
        with open(os.path.join(self.jsondir, filename), 'w') as outfile:
            json.dump(onnx_json, outfile, indent=2)
        self.__info(filename, self.jsondir)

    def print_json_from_DORY_graph(self, filename, graph):
        filename += '.json'
        # Logging function to report exported graph of PULP
        dict_graph = {"graph": [node.export_to_dict() for node in graph]}
        with open(os.path.join(self.jsondir, filename), 'w') as outfile:
            json.dump(dict_graph, outfile, indent=2)
        self.__info(filename, self.jsondir)

    def print_onnx_from_DORY_graph(self, filename, graph):
        filename += '.onnx'
        dict_graph = {"producerName": "DORY", "producerVersion": "",
                      "graph": {"node": [node.export_to_onnx() for node in graph]}}
        onnx_str = json.dumps(dict_graph)
        convert_model = Parse(onnx_str, onnx.ModelProto())
        onnx.save_model(convert_model, os.path.join(self.onnxdir, filename))
        self.__info(filename, self.onnxdir)
