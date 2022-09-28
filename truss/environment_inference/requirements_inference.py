import inspect
import types
from itertools import dropwhile
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
IGNORED_PACKAGES = {"pip", "truss", "pluggy", "pytest", "py"}

TOP_LEVEL_NAMESPACES_TO_DROP_FOR_INFERENCE = ["truss", "baseten"]


def infer_deps(must_include_deps: Set[str] = None) -> Set[str]:
    """Infers the depedencies based on imports into the global namespace

    Args:
        must_include_deps (Set, optional):  The set of package names that
                                            must necessarily be imported.
                                            Defaults to None.

    Returns:
        Set[str]: set of required python requirements, including versions. E.g. `{"xgboost==1.6.1"}`
    """

    # Find the stack frame that likely has the relevant global inputs
    stack = inspect.stack()
    try:
        relevant_stack = _filter_truss_frames(stack)
    except StopIteration:
        return set()

    if not must_include_deps:
        must_include_deps = set()

    pkg_candidates = _extract_packages_from_frame(relevant_stack[0].frame)
    imports = must_include_deps.union(pkg_candidates)
    requirements = set([])

    # Must refresh working set manually to get latest installed
    pkg_resources.working_set = (
        WorkingSet._build_master()  # pylint: disable=protected-access
    )

    # Cross-check the names of installed packages vs. imported packages to get versions
    for pkg_in_frame in pkg_resources.working_set:
        if (
            pkg_in_frame.project_name in imports
            and pkg_in_frame.project_name not in IGNORED_PACKAGES
        ):
            requirements.add(f"{pkg_in_frame.project_name}=={pkg_in_frame.version}")
            # Remove the package from imports as it was added into requirements
            imports.remove(pkg_in_frame.project_name)

    # Add the must include deps not found in frame to requirements
    deps_not_found_in_frame = imports.intersection(must_include_deps)
    requirements = requirements.union(deps_not_found_in_frame)

    return requirements


def _filter_truss_frames(stack_frames):
    def is_truss_invocation_frame(stack_frame):
        module = inspect.getmodule(stack_frame.frame)
        if not module:
            return False

        module_name = module.__name__
        for namespace in TOP_LEVEL_NAMESPACES_TO_DROP_FOR_INFERENCE:
            if module_name.startswith(f"{namespace}."):
                return True
        return False

    return list(dropwhile(is_truss_invocation_frame, stack_frames))


def _extract_packages_from_frame(frame) -> Set[str]:
    candidate_symbols = {**frame.f_globals, **frame.f_locals}

    pkg_names = set()
    for name, val in candidate_symbols.items():
        if name.startswith("__"):
            continue

        if isinstance(val, types.ModuleType):
            # Split ensures you get root package,
            # not just imported function
            pkg_name = val.__name__.split(".")[0]
        elif hasattr(val, "__module__"):
            pkg_name = val.__module__.split(".")[0]
        else:
            continue

        if pkg_name in POORLY_NAMED_PACKAGES:
            pkg_name = POORLY_NAMED_PACKAGES[pkg_name]

        pkg_names.add(pkg_name)

    return pkg_names
