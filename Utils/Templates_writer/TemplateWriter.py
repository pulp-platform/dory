from mako.template import Template
import os


class TemplateWriter:
    def __init__(self, node):
        pass

    # Assumes tmpl_files is a list of 2 files where one ends with .h and the other .c
    def write(self, tmpl_files, out_dir):
        for tmpl_file in tmpl_files:
            tmpl_file_name = os.path.basename(tmpl_file)
            if tmpl_file_name.endswith('.h'):
                file_dir = 'inc'
                out_file_name = self.func_name + '.h'
            elif tmpl_file_name.endswith('.c'):
                file_dir = 'src'
                out_file_name = self.func_name + '.c'
            else:
                file_dir = '.'
                out_file_name = self.func_name
            out_file = os.path.join(out_dir, file_dir, out_file_name)
            tmpl = Template(filename=tmpl_file)
            rendered = tmpl.render(**self.__dict__)
            with open(out_file, 'w') as f:
                f.write(rendered)
