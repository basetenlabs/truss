import base64
import errno
import logging
import os
from pathlib import Path

# TODO(marius/TaT): remove try-except after TaT.
# TODO(pankaj) In desparate need of refactoring into separate library
try:
    from helpers.custom_types import Action, Patch
except ModuleNotFoundError as exc:
    logging.debug(f"Importing helpers from truss core, caused by: {exc}")
    from truss.templates.control.control.helpers.custom_types import Action, Patch


def _resolve_within(base_dir: Path, relative_path: str) -> Path:
    """Resolve ``relative_path`` against ``base_dir`` and ensure the result
    stays inside ``base_dir``.

    The patch path comes from the request body of the control server's
    ``/control/patch`` endpoint, so a value such as ``../../../etc/cron.d/x``
    (or an absolute path, which ``/`` join would leave unchanged) must not be
    allowed to write or delete files outside the target directory.
    """
    base_resolved = base_dir.resolve()
    filepath = (base_dir / relative_path).resolve()
    if not filepath.is_relative_to(base_resolved):
        raise ValueError(
            f"Invalid patch path {relative_path!r}: resolves outside the "
            f"target directory {base_resolved}."
        )
    return filepath


def apply_code_patch(relative_dir: Path, patch: Patch, logger: logging.Logger):
    logger.debug(f"Applying code patch {patch.to_dict()}")
    filepath: Path = _resolve_within(relative_dir, patch.path)
    action = patch.action

    if action in [Action.ADD, Action.UPDATE]:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        action_log = "Adding" if action == Action.ADD else "Updating"
        logger.info(f"{action_log} file {filepath}")
        if patch.content_bytes is not None:
            # Binary file — decode base64 and write as bytes
            filepath.write_bytes(base64.b64decode(patch.content_bytes, validate=True))
        else:
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
                if (
                    e.errno == errno.ENOTEMPTY
                ):  # Directory not empty (actual number is different in Linux vs Mac)
                    pass
                else:
                    raise
    else:
        raise ValueError(f"Unknown patch action {action}")
