import logging
import os
from pathlib import Path

# TODO(pankaj) In desparate need of refactoring into separate library
try:
    from helpers.types import Action, Patch
except ModuleNotFoundError as exc:
    logging.debug(f"Importing helpers from truss core, caused by: {exc}")
    from truss.templates.control.control.helpers.types import Action, Patch


def apply_code_patch(
    relative_dir: Path,
    patch: Patch,
    logger: logging.Logger,
):
    logger.debug(f"Applying code patch {patch.to_dict()}")
    filepath: Path = relative_dir / patch.path
    action = patch.action

    if action in [Action.ADD, Action.UPDATE]:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        action_log = "Adding" if action == Action.ADD else "Updating"
        logger.info(f"{action_log} file {filepath}")
        with filepath.open("w") as file:
            content = patch.content
            if content is None:
                raise ValueError(
                    "Invalid patch: content of a file update patch should not be None."
                )
            file.write(content)

    elif action == Action.REMOVE:
        if not filepath.exists():
            logger.warning(f"Could not delete file {filepath}: not found.")
        elif filepath.is_file():
            logger.info(f"Deleting file {filepath}")
            filepath.unlink()
            # attempt to recursively remove potentially empty directories, if applicable
            # os.removedirs raises OSError with errno 39 when this process encounters a non-empty dir
            try:
                os.removedirs(filepath.parent)
            except OSError as e:
                if e.errno == 39:  # Directory not empty
                    pass
                else:
                    raise
    else:
        raise ValueError(f"Unknown patch action {action}")
