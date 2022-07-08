import os

from .TemplateWriter import TemplateWriter
from . import writer_utils as utils


class MiscTemplateWriter(TemplateWriter):
    def __init__(self, graph, hw_desc, conf, verbose_level, perf_layer):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
        tmpldir = os.path.join(root, 'Hardware-targets', hw_desc['name'], 'Templates')
        super().__init__(tmpldir)

        weights_hex_files = [node.name + "_weights.hex" for node in graph if node.has_weights()]

        self.tk['verbose'] = 'Check' in verbose_level
        self.tk['weights_number'] = sum([1 for node in graph if node.has_weights()])
        self.tk['verbose_level'] = verbose_level
        self.tk['performance'] = perf_layer
        self.tk['l1_buffer'] = hw_desc["memory"]["L1"]["dimension"] \
                               - hw_desc["HW specific parameters"]["accelerator core0 stack"] \
                               - 7 * hw_desc["HW specific parameters"]["accelerator core1-7 stack"]
        self.tk['master_stack'] = hw_desc["HW specific parameters"]["accelerator core0 stack"]
        self.tk['slave_stack'] = hw_desc["HW specific parameters"]["accelerator core1-7 stack"]
        self.tk['l2_buffer_size'] = hw_desc["memory"]["L2"]["dimension"] - conf["code reserved space"]
        self.tk['MACs'] = sum([node.MACs for node in graph])
        self.tk['layers_w'] = weights_hex_files
        self.tk['files_list'] = utils.print_file_list(weights_hex_files)
        self.tk['fc_frequency'] = hw_desc["core frequency"]
        self.tk['cl_frequency'] = hw_desc["accelerator frequency"]
        self.tk['sdk'] = hw_desc["software development kit"]["name"]
        self.tk['list_h'] = [node.name + '.h' for node in graph]
        self.tk['func_name'] = [node.name for node in graph]
        self.tk['verbose_log'] = "".join([f'// {k:<30} {v}' for k, v in self.tk.items()])
        self.tk['DORY_HW_graph'] = graph

