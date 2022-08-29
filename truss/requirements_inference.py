import inspect
import types
from typing import Set

import pkg_resources
from pkg_resources import WorkingSet

# Some packages are weird and have different
# imported names vs. system/pip names. Unfortunately,
# there is no systematic way to get pip names from
# a package's imported name. You'll have to add
# exceptions to this list manually!
POORLY_NAMED_PACKAGES = {"PIL": "Pillow", "sklearn": "scikit-learn"}

# We don't want a few foundation packages
IGNORED_PACKAGES = {"pip", "truss"}


def infer_deps(
    root_fn_name: str = "mk_truss", must_include_deps: Set = None
) -> Set[str]:
    """Infers the depedencies based on imports into the global namespace

    Args:
        root_fn_name (str, optional):   The name of the function that's called
                                        where the global namespace is relevant.
                                        Defaults to "mk_truss".
        must_include_deps (Set, optional):  The set of package names that
                                            must necessarily be imported.
                                            Defaults to None.

    Returns:
        Set[str]: set of required python requirements, including versions. E.g. `{"xgboost==1.6.1"}`
    """

    # Find the stack frame that likely has the relevant global inputs
    try:
        relevant_stack = next(
            filter(lambda s: s.function == root_fn_name, inspect.stack())
        )
    except StopIteration:
        return set()

    global_state_of_caller = relevant_stack.frame.f_globals
    imports = must_include_deps.copy() if must_include_deps else set()

    for name, val in global_state_of_caller.items():
        if isinstance(val, types.ModuleType):
            # Split ensures you get root package,
            # not just imported function
            name = val.__name__.split(".")[0]

        elif isinstance(val, type):
            name = val.__module__.split(".")[0]

        if name in POORLY_NAMED_PACKAGES:
            name = POORLY_NAMED_PACKAGES[name]

        imports.add(name)

    requirements = set([])

    # Must refresh working set manually to get latest installed
    pkg_resources.working_set = (
        WorkingSet._build_master()  # pylint: disable=protected-access
    )

    # Cross-check the names of installed packages vs. imported packages to get versions
    for m in pkg_resources.working_set:
        if m.project_name in imports and m.project_name not in IGNORED_PACKAGES:
            requirements.add(f"{m.project_name}=={m.version}")

    return requirements
