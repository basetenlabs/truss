import os
from typing import Dict, Optional, Set


class override_env_vars:
    """A context manager for temporarily overwriting environment variables.

    Usage:
        with override_env_vars({'API_KEY': 'test_key', 'DEBUG': 'true'}):
            # Environment variables are modified here
            ...
        # Original environment is restored here
    """

    def __init__(
        self,
        env_vars: Optional[Dict[str, str]] = None,
        deleted_vars: Optional[Set[str]] = None,
    ):
        """
        Args:
            env_vars: Dictionary of environment variables to set
            deleted_vars: Set of environment variables to delete
        """
        self.env_vars: Dict[str, str] = env_vars or dict()
        self.deleted_vars: Set[str] = deleted_vars or set()
        self.original_vars: Dict[str, Optional[str]] = {}

    def __enter__(self):
        all_keys = set(self.env_vars.keys()) | self.deleted_vars
        for key in all_keys:
            self.original_vars[key] = os.environ.get(key)

        for key, value in self.env_vars.items():
            os.environ[key] = value

        for key in self.deleted_vars:
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
