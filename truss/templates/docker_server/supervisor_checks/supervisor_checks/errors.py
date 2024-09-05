"""Error classes.
"""

__author__ = 'vovanec@gmail.com'


class InvalidCheckConfig(ValueError):
    """Raised when invalid configuration dictionary passed to check module.
    """

    pass


class InvalidPortSpec(InvalidCheckConfig):
    """Raised when invalid port specification was provided in config.
    """

    pass
