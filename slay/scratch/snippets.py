import ast

import libcst as cst


def _remove_main(source_tree: ast.Module):
    new_body = [
        node
        for node in source_tree.body
        if not (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
            and isinstance(node.test.comparators[0], ast.Constant)
            and node.test.comparators[0].value == "__main__"
        )
    ]
    source_tree.body = new_body

    # def __init__(
    #     self,
    #     config,
    #     data_generator: GenerateData = GenerateData(),
    #     splitter: shared_processor.SplitText = shared_processor.SplitText(),
    #     text_to_num: TextToNum = TextToNum(),
    # ) -> None:
    #     super().__init__(config)
    #     self._data_generator = data_generator
    #     self._data_splitter = splitter
    #     self._text_to_num = text_to_num

    #     self._data_generator = stubs.GenerateData(config)
    #     self._data_splitter = stubs.SplitText(config)
    #     self._text_to_num = stubs.TextToNum(config)


src = """
import stubs

class Workflow(slay.ProcessorBase):
    default_config = slay.Config(image=IMAGE_COMMON)

    def __init__(
        self,
        config: slay.Config = slay.provide_config(),
        data_generator: GenerateData = slay.provide(GenerateData),
        splitter: shared_processor.SplitText = slay.provide(shared_processor.SplitText),
        text_to_num: TextToNum = slay.provide(TextToNum),
    ) -> None:
        super().__init__(config)
        self._data_generator = data_generator
        self._data_splitter = splitter
        self._text_to_num = text_to_num
"""


new_instantiations = {
    "data_generator": {"class": "stubs_GenerateData", "argument": "config"},
    "data_splitter": {"class": "stubs_SplitText", "argument": "config"},
    "text_to_num": {"class": "stubs_TextToNum", "argument": "config"},
}


# class RewriteInitVisitor(libcst.CSTVisitor):
#     def __init__(self, new_instantiations):
#         super().__init__()
#         self.new_instantiations = new_instantiations

#     def visit_FunctionDef(self, node: libcst.FunctionDef):
#         if node.name.value == "__init__":
#             # Process parameters (excluding 'self' and 'config')
#             params_to_remove = [param for param in node.params.params[2:]]
#             new_params = node.params.with_changes(params=node.params.params[:2])

#             # Create new instantiation statements
#             new_statements = []
#             for param in params_to_remove:
#                 print(param.name.value)
#                 if param.name.value in self.new_instantiations:
#                     instantiation = self.new_instantiations[param.name.value]
#                     new_statement = libcst.SimpleStatementLine(
#                         body=[
#                             libcst.Assign(
#                                 targets=[libcst.Name(value=param.name.value)],
#                                 value=libcst.Call(
#                                     func=libcst.Name(value=instantiation["class"]),
#                                     args=[
#                                         libcst.Arg(
#                                             value=libcst.Name(
#                                                 value=instantiation["argument"]
#                                             )
#                                         )
#                                     ],
#                                 ),
#                             )
#                         ]
#                     )
#                     new_statements.append(new_statement)

#             print(new_statements)

#             # Replace the parameters and add new instantiation statements to the body
#             new_body = list(node.body.body) + new_statements
#             new = node.with_changes(
#                 params=new_params, body=node.body.with_changes(body=new_body)
#             )
#         return True

# module = libcst.parse_module(src)
# # module.visit(RewriteInitVisitor(new_instantiations=new_instantiations))
# print(module.code)


class InitRewriter(cst.CSTTransformer):
    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        keep_params = ["self", "config"]
        if updated_node.name.value == "__init__":
            new_params = updated_node.params.with_changes(
                params=[
                    param
                    for param in updated_node.params.params
                    if param.name.value in keep_params
                ]
            )

            # Create new statements for the method body
            new_body = updated_node.body.with_changes(
                body=[
                    cst.parse_statement("data_generator = stubs.GenerateData(config)"),
                    cst.parse_statement("data_splitter = stubs.SplitText(config)"),
                    cst.parse_statement("text_to_num = stubs.TextToNum(config)"),
                    *updated_node.body.body,
                ]
            )

            return updated_node.with_changes(
                params=new_params,
                body=new_body,
            )
        return updated_node


tree = cst.parse_module(src)
modified_tree = tree.visit(InitRewriter())
print(modified_tree.code)
