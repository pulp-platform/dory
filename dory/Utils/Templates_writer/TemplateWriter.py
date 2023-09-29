from mako.template import Template
from typing import Dict, Any


class TemplateWriter:

    @staticmethod
    def write(templateKeywordDict: Dict[str, Any], templateMapping: Dict[str, str]):
        for dest, template in templateMapping.items():
            # Flag strict_undefined enables reporting of names of the undefined variables (very useful)
            # but expects all the variables to be defined which is not our case. We do some conditional
            # rendering with flags like has_bias, has_batchnorm, etc. so some variables raise undefined
            # error even though they wouldn't be used. For now keep strict_undefined as False, until
            # the keyword generation gets changed to generate all the kewywords.
            render = Template(filename=template, strict_undefined=False).render(**templateKeywordDict)
            assert isinstance(render, str)
            with open(dest, "w") as fp:
                fp.write(render)
