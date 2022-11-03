# C_Parser.py
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

from dory.Hardware_targets.PULP.Common import C_Parser_PULP
import os

class C_Parser(C_Parser_PULP):
    def __init__(self, *args, **kwargs):
        super(C_Parser, self).__init__(*args, **kwargs)
        if self.precision_library == "mixed-hw":
            assert False, "optional='mixed-hw' not compatible with GAP8!"

    def get_file_path(self):
        return "/".join(os.path.realpath(__file__).split("/")[:-1])
