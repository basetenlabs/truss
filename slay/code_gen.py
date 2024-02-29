import ast
import functools
import logging

import black
import httpx
import isort
from slay import utils


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


def edit_model_file(file_path):
    with open(file_path, "r", encoding="utf-8") as source_file:
        source_code = source_file.read()
    source_tree = ast.parse(source_code)

    _remove_main(source_tree)

    modified_source_code = ast.unparse(source_tree)
    # Format and clean the file.
    with utils.log_level(logging.INFO):
        formatted_code = black.format_file_contents(
            modified_source_code, fast=False, mode=black.FileMode()
        )
    with open(file_path, "w", encoding="utf-8") as modified_file:
        modified_file.write(formatted_code)
    # isort.file(file_path)
