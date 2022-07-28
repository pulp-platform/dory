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
from google.protobuf.json_format import MessageToJson
from google.protobuf.json_format import Parse

class Printer():
    def __init__(self, folder):
        self.folder = folder
        os.system('rm -rf {}'.format(self.folder))
        os.system('mkdir -p {}'.format(os.path.join(self.folder,"json_files")))
        os.system('mkdir -p {}'.format(os.path.join(self.folder,"onnx_files")))

    def print_onnx(self, name_file, graph):
        onnx.save_model(graph, "{}/{}.onnx".format(os.path.join(self.folder,"onnx_files"),name_file))
        print("Creating {}.onnx in {}/". format(name_file, os.path.join(self.folder,"onnx_files")))

    def print_json(self, name_file, graph):
        s = MessageToJson(graph )
        onnx_json = json.loads(s)
        out_file = open("{}/{}.json".format(os.path.join(self.folder,"json_files"),name_file), "w") 
        json.dump(onnx_json, out_file, indent = 2) 
        out_file.close() 
        print("Creating {}.json in {}/". format(name_file, os.path.join(self.folder,"json_files")))


    def print_json_from_DORY_graph(self, name_file, graph):
        # Logging function to report exported graph of PULP
        dict_graph = {}
        dict_graph["graph"] = []
        for i,nodes in enumerate(graph):
            dict_graph["graph"].append(nodes.export_to_dict()) 
        with open("{}/{}.json".format(os.path.join(self.folder,"json_files"),name_file), "w") as outfile:
            json.dump(dict_graph, outfile, indent=2)
        print("Creating {}.json in {}/". format(name_file, os.path.join(self.folder,"json_files")))

    def print_onnx_from_DORY_graph(self, name_file, graph):

        dict_graph = {}
        dict_graph["producerName"] = "DORY"
        dict_graph["producerVersion"] = ""
        dict_graph["graph"] = {}
        dict_graph["graph"]["node"] = []
        for i,nodes in enumerate(graph):
            dict_graph["graph"]["node"].append(nodes.export_to_onnx()) 
        onnx_str = json.dumps(dict_graph)

        convert_model = Parse(onnx_str, onnx.ModelProto())
        onnx.save_model(convert_model,"{}/{}.onnx".format(os.path.join(self.folder,"onnx_files"),name_file))
        print("Creating {}.onnx in {}/". format(name_file, os.path.join(self.folder,"onnx_files")))