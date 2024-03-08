import astroid
from slay import definitions, utils


def _get_ast(src_file: str):
    with open(src_file, "r") as file:
        source_code = file.read()

    return astroid.parse(source_code)


def is_main_section(node):
    # Check if the node is in the `if __name__ == '__main__':` block
    current = node
    while current:
        if isinstance(current, astroid.If):
            test = current.test
            if (
                isinstance(test, astroid.Compare)
                and isinstance(test.left, astroid.Name)
                and test.left.name == "__name__"
                and any(
                    isinstance(right, astroid.Const) and right.value == "__main__"
                    for right in test.ops[0][1]
                )
            ):
                return True
            else:
                raise ValueError("No if-blocks allowed.")
        current = current.parent

    return False


def get_module_namse(module_ast: astroid.Module):
    top_level_names = []

    for node in module_ast.body:
        if isinstance(
            node, (astroid.Assign, astroid.FunctionDef, astroid.ClassDef)
        ) and not is_main_section(node):
            if isinstance(node, astroid.Assign):
                for target in node.targets:
                    if isinstance(target, astroid.AssignName):
                        top_level_names.append(target.name)
                    elif isinstance(target, astroid.Tuple):
                        top_level_names.extend(
                            [
                                name.name
                                for name in target.elts
                                if isinstance(name, astroid.AssignName)
                            ]
                        )
            else:
                top_level_names.append(node.name)

    return top_level_names


def analyze(processor_descriptor: definitions.ProcessorAPIDescriptor):
    module_ast = _get_ast(processor_descriptor.src_file)

    module_names = get_module_namse(module_ast)
    print(module_names)
    processor_ast = utils.expect_one(
        node
        for node in module_ast.body
        if isinstance(node, astroid.ClassDef)
        and node.name == processor_descriptor.cls_name
    )

    def collect_accessed_names(node: astroid.NodeNG, accessed_names):
        if isinstance(node, astroid.Name) and node.ctx == "load":
            accessed_names.add(node.name)
        elif isinstance(node, astroid.Attribute) and isinstance(node.ctx, astroid.Load):
            # For attribute access, you might want to collect the attribute name,
            # or the full path (e.g., obj.attr). Uncomment the appropriate line below.
            accessed_names.add(node.attrname)  # Collect only the attribute name.
            # accessed_names.add(f"{node.expr.as_string()}.{node.attrname}")  # Collect full path.

        # Recursively inspect child nodes
        for child in node.get_children():
            collect_accessed_names(child, accessed_names)

    accessed_names = set()
    collect_accessed_names(processor_ast, accessed_names)

    # Exclude variables accessed in __init__'s signature
    def exclude_init_signature(class_def: astroid.ClassDef):
        init_method = next(
            (m for m in class_def.methods() if m.name == "__init__"), None
        )
        if init_method is not None:
            for arg in init_method.args.args:
                if isinstance(arg, astroid.AssignName) and arg.name in module_names:
                    print(f"Discarding {arg.name}")
                    accessed_names.discard(arg.name)

    exclude_init_signature(processor_ast)
    print(accessed_names)


if __name__ == "__main__":

    class Workflow(definitions.ABCProcessor):
        ...

    test_class = Workflow

    descr = definitions.ProcessorAPIDescriptor(
        processor_cls=test_class,
        src_file="/home/marius-baseten/workbench/truss/example_workflow/workflow.py",
        depdendencies={},
        endpoints=[],
    )

    analyze(descr)
