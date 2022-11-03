from dory.Hardware_targets.PULP.Common import onnx_manager_PULP
from dory.Hardware_targets.PULP.PULP_gvsoc.HW_Pattern_rewriter import Pattern_rewriter
from dory.Hardware_targets.PULP.PULP_gvsoc.Tiler import Tiler
import os

class onnx_manager(onnx_manager_PULP):
    def get_file_path(self):
        return "/".join(os.path.realpath(__file__).split("/")[:-1])

    def get_pattern_rewriter(self):
        return Pattern_rewriter

    def get_tiler(self):
        return Tiler
