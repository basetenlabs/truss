from docstring_parser import parse

path = "truss/truss_config.py"

to_document = "TrussConfig"

with open(path, "r") as f:
    docstring = f.read()

docstring = docstring.split(f"class {to_document}:\n")[1].split('"""')[1]

docstring = parse(docstring)

mdx = ""

for param in docstring.params:
    mdx += f'<ParamField body="{param.arg_name}" type="{param.type_name}">\n{param.description}\n</ParamField>\n'

out = open("../docs/_snippets/config_params.mdx", "w")
out.write(mdx)
out.close()
