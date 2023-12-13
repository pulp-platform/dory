from dory.Hardware_targets.PULP.GAP9_NE16.HW_Parser import onnx_manager as onnx_manager_gap9_ne16
from .HW_Pattern_rewriter import Pattern_rewriter
from dory.Hardware_targets.PULP.GAP9_NE16.Tiler.tiler import Tiler_GAP9 as Tiler
import os

class onnx_manager(onnx_manager_gap9_ne16):
    def get_file_path(self):
        return "/".join(os.path.realpath(__file__).split("/")[:-1])

    def get_pattern_rewriter(self):
        return Pattern_rewriter

    def get_tiler(self):
        return Tiler
