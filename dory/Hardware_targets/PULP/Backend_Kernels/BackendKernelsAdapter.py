from typing import Literal, List, Union
import os

from dory.Parsers.HW_node import HW_node


class BackendKernelsAdapter:
    def __init__(self, dirname: str, node: HW_node):
        self.dirname = dirname
        self._check_valid_node(node)
        self.node = node

    def _check_valid_node(self, node: HW_node) -> None:
        _ = node
        raise NotImplementedError()

    def _root_dir(self) -> Union[os.PathLike, str]:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), self.dirname)

    def _get_src_files(self) -> List[str]:
        raise NotImplementedError()

    def _get_inc_files(self) -> List[str]:
        raise NotImplementedError()

    @staticmethod
    def _filter(files: List[str], extension: str):
        return [file for file in files if file.endswith(extension)]

    def get_src_files(self) -> List[str]:
        return BackendKernelsAdapter._filter(self._get_src_files(), ".c")

    def get_inc_files(self) -> List[str]:
        return BackendKernelsAdapter._filter(self._get_inc_files(), ".h")


class PulpNNAdapter(BackendKernelsAdapter):
    def __init__(self, dirname: str, node: HW_node, constant_bits: int):
        super().__init__(dirname, node)
        self.constant_bits = constant_bits

    def _check_valid_node(self, node: HW_node) -> None:
        _ = node
        assert True

    def _src_dir(self) -> Union[os.PathLike, str]:
        return os.path.join(self._root_dir(), "{}bit/src".format(self.constant_bits))

    def _inc_dir(self) -> Union[os.PathLike, str]:
        return os.path.join(
            self._root_dir(), "{}bit/include".format(self.constant_bits)
        )

    def _get_src_files(self) -> List[str]:
        path, _, src_files = next(os.walk(self._src_dir()))
        return [os.path.join(path, file) for file in src_files]

    def _get_inc_files(self) -> List[str]:
        path, _, inc_files = next(os.walk(self._inc_dir()))
        return [os.path.join(path, file) for file in inc_files]


class PulpMixedAdapter(PulpNNAdapter):
    def __init__(
        self,
        dirname: str,
        node: HW_node,
        constant_bits: int,
        _type: Literal["hw", "sw"],
    ):
        self._type = _type
        if _type == "hw":
            dirname = os.path.join(dirname, "XpulpNN")
        elif _type == "sw":
            dirname = os.path.join(dirname, "XpulpV2")
        else:
            assert (
                False
            ), "ERROR: Unrecognised PulpMixed kernel library type: {}".format(_type)
        super().__init__(dirname, node, constant_bits)

    def _check_valid_node(self, node: HW_node) -> None:
        _ = node
        assert True

    def _get_src_files(self) -> List[str]:
        in_bits = str(self.node.get_parameter("input_activation_bits"))
        out_bits = str(self.node.get_parameter("output_activation_bits"))
        in_type = self.node.get_parameter("input_activation_type")[0]
        out_type = self.node.get_parameter("output_activation_type")[0]

        out = "_" + out_type + out_bits
        in_out = "_" + in_type + in_bits + out
        maybe_x = "x" if self._type == "hw" else ""

        src_files = []

        assert isinstance(self.node.name, str)
        assert isinstance(self.node.op_type, str)

        if "Addition" in self.node.name:
            in1_in2_out = (
                "_"
                + in_type
                + in_bits
                + "_"
                + self.node.get_parameter("second_input_activation_type")[0]
                + str(self.node.get_parameter("second_input_activation_bits"))
                + "_"
                + out_type
                + out_bits
            )
            src_files.append(f"Add/{maybe_x}pulp_nn_add{in1_in2_out}.c")
        elif "Pool" in self.node.name and "Max" in self.node.op_type:
            src_files.append(f"Pooling/MaxPool/{maybe_x}pulp_nn_maxpool{out}.c")
        elif "Pool" in self.node.name and (
            "Avg" in self.node.op_type or "Average" in self.node.op_type
        ):
            src_files.append(f"Pooling/AvgPool/{maybe_x}pulp_nn_avgpool{in_out}.c")

        in_out_weights = (
            "_"
            + in_type
            + in_bits
            + "_"
            + out_type
            + out_bits
            + "_"
            + self.node.get_parameter("weight_type")[0]
            + str(self.node.get_parameter("weight_bits"))
        )
        if "Conv" in self.node.name and self.node.group > 1:
            src_files.append(f"Depthwise/{maybe_x}pulp_nn_depthwise{in_out_weights}.c")
        elif "Conv" in self.node.name and self.node.group == 1:
            if self.node.conv1d and self._type == "hw":
                src_files.append(f"Convolution/xpulp_nn_conv1d{in_out_weights}.c")
            else:
                src_files.append(f"Convolution/{maybe_x}pulp_nn_conv{in_out_weights}.c")
        elif (
            "FullyConnected" in self.node.name
            and self.node.output_activation_bits == 32
        ):
            src_files.append(f"LinearNoQuant/{maybe_x}pulp_nn_linear{in_out_weights}.c")
        elif "FullyConnected" in self.node.name:
            src_files.append(f"LinearQuant/{maybe_x}pulp_nn_linear{in_out_weights}.c")

        if (
            "Conv" in self.node.name or "FullyConnected" in self.node.name
        ) and self.node.get_parameter("output_activation_bits") != 32:
            in_bits_matmul = "8" if self._type == "sw" else str(in_bits)
            in_out_weights = (
                "_"
                + in_type
                + in_bits_matmul
                + "_"
                + out_type
                + out_bits
                + "_"
                + self.node.get_parameter("weight_type")[0]
                + str(self.node.get_parameter("weight_bits"))
            )
            src_files.append(
                f"MatrixMultiplication/{maybe_x}pulp_nn_matmul{in_out_weights}.c"
            )

        src_files = [os.path.join(self._src_dir(), file) for file in src_files]

        return src_files
