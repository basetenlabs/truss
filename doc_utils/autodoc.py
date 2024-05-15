#  type: ignore
from docstring_parser import parse

path = "../truss/truss_config.py"

to_document = "TrussConfig"

with open(path, "r") as f:
    docstring = f.read()

docstring = docstring.split(f"class {to_document}:\n")[1].split('"""')[1]

docs = parse(docstring)

mdx = ""

for param in docs.params:
    mdx += f'<ParamField body="{param.arg_name}" type="{param.type_name}">\n{param.description}\n</ParamField>\n'

with open("../docs/snippets/config-params.mdx", "w") as out:
    out.write(mdx)
