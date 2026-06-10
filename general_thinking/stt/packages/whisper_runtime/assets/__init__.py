"""
Package assets including model files and configurations.
"""

import os
from pathlib import Path

ASSETS_DIR = Path(__file__).parent
MEL_FILTERS_PATH = ASSETS_DIR / "mel_filters.npz"
LANG_CODES_PATH = ASSETS_DIR / "lang_codes.txt"
LANG_MAPPING_PATH = ASSETS_DIR / "lang_code_mapping.json"


def get_asset_path(filename: str) -> Path:
    """Get the full path to an asset file."""
    return ASSETS_DIR / filename


__all__ = [
    "ASSETS_DIR",
    "MEL_FILTERS_PATH",
    "LANG_CODES_PATH",
    "LANG_MAPPING_PATH",
    "get_asset_path",
]
