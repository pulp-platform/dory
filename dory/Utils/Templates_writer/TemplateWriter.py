from mako.template import Template
from typing import Dict, Any


class TemplateWriter:

    @staticmethod
    def write(templateKeywordDict: Dict[str, Any], templateMapping: Dict[str, str]):
        for dest, template in templateMapping.items():
            render = Template(filename=template).render(**templateKeywordDict)
            assert isinstance(render, str)
            with open(dest, "w") as fp:
                fp.write(render)
