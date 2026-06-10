import fcntl
import json
import logging
import os
import shutil
import tarfile
import time
from contextlib import contextmanager
from pathlib import Path

import torch

DEVICE_NAME = (
    torch.cuda.get_device_name().upper().replace(" ", "_") if torch.cuda.is_available() else "CPU"
)
B10FS_CACHE_DIR = Path(f"/cache/org/asr_streaming_compile_cache/{DEVICE_NAME}")
TORCH_COMPILE_CACHE_DIR = Path("/tmp/torchinductor_root")

IS_CACHE_SAVED = False


def is_cache_saved() -> bool:
    """Check if the compile cache is already saved by verifying file existence."""
    global IS_CACHE_SAVED
    if IS_CACHE_SAVED:
        # in-memory variable is true, so we don't need to check the file system
        return True

    # need to see if cache has now been saved since we last checked
    IS_CACHE_SAVED = os.path.exists(B10FS_CACHE_DIR / "cache_info.json") and os.path.exists(
        B10FS_CACHE_DIR / "compile_cache.tar.gz"
    )
    return IS_CACHE_SAVED


logger = logging.getLogger(__name__)


@contextmanager
def file_lock(file_path: Path):
    """Context manager for file locking to prevent race conditions."""
    lock_file = file_path.parent / f"{file_path.name}.lock"
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    with open(lock_file, "w") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            # Clean up lock file if it exists
            try:
                lock_file.unlink()
            except FileNotFoundError:
                pass


def load_compile_cache() -> bool:
    """
    Load the compile cache from b10fs.
    Returns False if the cache does not exist. True if cache was successfully loaded.
    """
    start_time = time.time()
    cache_tar_file = B10FS_CACHE_DIR / "compile_cache.tar.gz"

    if TORCH_COMPILE_CACHE_DIR.exists() and len(os.listdir(TORCH_COMPILE_CACHE_DIR)) > 0:
        logger.info(
            f"[Torch Compile Cache] Torch compile cache already exists at {TORCH_COMPILE_CACHE_DIR}."
        )
        return True

    if not cache_tar_file.exists():
        logger.info(f"[Torch Compile Cache] No torch compile cache found at {cache_tar_file}.")
        return False

    with file_lock(cache_tar_file):
        TORCH_COMPILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        with tarfile.open(cache_tar_file, "r:gz") as tar:
            # Extract with safety filter and to the target directory
            tar.extractall(TORCH_COMPILE_CACHE_DIR, filter="data")

    end_time = time.time()
    logger.info(
        f"[Torch Compile Cache] Loaded torch compile cache from {TORCH_COMPILE_CACHE_DIR} in {end_time - start_time:.2f} seconds."
    )
    return True


def save_compile_cache() -> dict | None:
    """
    Save the compile cache for this model to b10fs.
    Returns None if the cache was not successfully saved. Returns the cache info if it was successfully saved.
    """
    start_time = time.time()

    if not TORCH_COMPILE_CACHE_DIR.exists():
        logger.info(
            f"[Torch Compile Cache] No torch compile cache found at {TORCH_COMPILE_CACHE_DIR}. Has the model been compiled yet?"
        )
        return None

    cache_tar_file = B10FS_CACHE_DIR / "compile_cache.tar.gz"
    cache_info_file = B10FS_CACHE_DIR / "cache_info.json"

    with file_lock(cache_tar_file):
        # Always check the file system state, not an in-memory variable
        if is_cache_saved():
            logger.info(
                f"[Torch Compile Cache] Torch compile cache already exists at {B10FS_CACHE_DIR}."
            )
            return None

        if B10FS_CACHE_DIR.exists():
            shutil.rmtree(B10FS_CACHE_DIR)

        B10FS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        with tarfile.open(cache_tar_file, "w:gz") as tar:
            # Add contents of torch compile directory with proper arcname to avoid nested paths
            for item in TORCH_COMPILE_CACHE_DIR.iterdir():
                tar.add(item, arcname=item.name)

        torch_cache_size = sum(
            f.stat().st_size for f in TORCH_COMPILE_CACHE_DIR.rglob("*") if f.is_file()
        )

        b10fs_cache_size = cache_tar_file.stat().st_size

        cache_info = {
            "b10fs_cache_size": b10fs_cache_size,
            "torch_cache_size": torch_cache_size,
            "duration": 0.0,
        }

        with open(cache_info_file, "w") as f:
            json.dump(cache_info, f)

    end_time = time.time()
    cache_info["duration"] = end_time - start_time
    logger.info(
        f"[Torch Compile Cache] Saved torch compile cache to {B10FS_CACHE_DIR} in {end_time - start_time:.2f} seconds."
    )

    return cache_info
