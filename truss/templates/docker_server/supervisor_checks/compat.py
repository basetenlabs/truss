"""Compatibility functions.
"""

__author__ = 'vovanec@gmail.com'

try:  # pragma: no cover
    from supervisor.compat import httplib
except ImportError:  # pragma: no cover
    import httplib

try:  # pragma: no cover
    from supervisor.compat import xmlrpclib
except ImportError:  # pragma: no cover
    import xmlrpclib
