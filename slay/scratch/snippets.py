import ast

import libcst as cst

# def _remove_main(source_tree: ast.Module):
#     new_body = [
#         node
#         for node in source_tree.body
#         if not (
#             isinstance(node, ast.If)
#             and isinstance(node.test, ast.Compare)
#             and isinstance(node.test.left, ast.Name)
#             and node.test.left.id == "__name__"
#             and isinstance(node.test.comparators[0], ast.Constant)
#             and node.test.comparators[0].value == "__main__"
#         )
#     ]
#     source_tree.body = new_body
