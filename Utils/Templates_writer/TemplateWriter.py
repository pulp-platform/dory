import os
import re
from mako.template import Template


class TemplateWriter:
    def __init__(self, tmpldir):
        self.tk = {}
        self.tmpldir = os.path.abspath(tmpldir)

    def set_var(self, name, val):
        self.tk[name] = val

    # Template file name should be of form tmpl_<filename>.<ext>
    # If called without outfiles, generates name from template file name
    def write(self, tmplfiles, dests):
        if not isinstance(tmplfiles, list):
            tmplfiles = [tmplfiles]

        if not isinstance(dests, list):
            dests = [dests]

        assert len(tmplfiles) == len(dests)

        regex = re.compile(r'tmpl_(.*)')

        for tmplfile, dest in zip(tmplfiles, dests):
            filename, ext = os.path.splitext(tmplfile)

            match = regex.fullmatch(filename)
            if match is None:
                print(f'Skipping template file {tmplfile}: not of form tmpl_<filename>[.c|.h]')
                continue

            template = Template(filename=os.path.join(self.tmpldir, tmplfile))
            rendered_template = template.render(**self.tk)

            outfile = os.path.join(dest, match.group(1) + ext) if os.path.isdir(dest) else dest
            with open(outfile, 'w') as file:
                file.write(rendered_template)
