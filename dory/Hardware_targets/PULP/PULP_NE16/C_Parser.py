#C_Parser.py
#Alessio Burrello <alessio.burrello@unibo.it>
#
#Copyright (C) 2019-2020 University of Bologna
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
from dory.Hardware_targets.PULP.GAP9_NE16.C_Parser import C_Parser as C_Parser_gap9_ne16
import os

class C_Parser(C_Parser_gap9_ne16):

    def get_file_path(self):
        return "/".join(os.path.realpath(__file__).split("/")[:-1])
