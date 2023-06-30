import logging
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
        logger.info(f"Updating file {filepath}")
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
        else:
            logger.info(f"Deleting file {filepath}")
            filepath.unlink()
    else:
        raise ValueError(f"Unknown patch action {action}")
