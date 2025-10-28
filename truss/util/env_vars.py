import os
from typing import Dict, Optional, Set


class modify_env_vars:
    """A context manager for temporarily overwriting environment variables.

    Usage:
        with modify_env_vars(overrides={'API_KEY': 'test_key', 'DEBUG': 'true'}, deletions={'AWS_CONFIG_FILE'}):
            # Environment variables are modified here
            ...
        # Original environment is restored here
    """

    def __init__(
        self,
        overrides: Optional[Dict[str, str]] = None,
        deletions: Optional[Set[str]] = None,
    ):
        """
        Args:
            overrides: Dictionary of environment variables to set
            deletions: Set of environment variables to delete
        """
        self.overrides: Dict[str, str] = overrides or dict()
        self.deletions: Set[str] = deletions or set()
        self.original_vars: Dict[str, Optional[str]] = {}

    def __enter__(self):
        all_keys = set(self.overrides.keys()) | self.deletions
        for key in all_keys:
            self.original_vars[key] = os.environ.get(key)

        for key, value in self.overrides.items():
            os.environ[key] = value

        for key in self.deletions:
            if key in os.environ:
                del os.environ[key]

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original environment
        for key, value in self.original_vars.items():
            if value is None:
                # Variable didn't exist originally
                if key in os.environ:
                    del os.environ[key]
            else:
                # Restore original value
                os.environ[key] = value
