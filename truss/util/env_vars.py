import os
from typing import Dict, Optional


class override_env_vars:
    """A context manager for temporarily overwriting environment variables.

    Usage:
        with override_env_vars({'API_KEY': 'test_key', 'DEBUG': 'true'}):
            # Environment variables are modified here
            ...
        # Original environment is restored here
    """

    def __init__(self, env_vars: Dict[str, str]):
        """
        Args:
            env_vars: Dictionary of environment variables to set
        """
        self.env_vars = env_vars
        self.original_vars: Dict[str, Optional[str]] = {}

    def __enter__(self):
        for key in self.env_vars:
            self.original_vars[key] = os.environ.get(key)

        for key, value in self.env_vars.items():
            os.environ[key] = value

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
