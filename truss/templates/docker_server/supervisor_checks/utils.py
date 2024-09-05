"""Utility functions.
"""


import contextlib
import functools
import re
import time

from supervisor_checks import errors

__author__ = 'vovanec@gmail.com'


RETRY_SLEEP_TIME = 3


class retry_errors(object):
    """Decorator to retry on errors.
    """

    def __init__(self, num_retries, log):

        self._num_retries = num_retries
        self._log = log

    def __call__(self, func):

        @functools.wraps(func)
        def wrap_it(*args, **kwargs):
            tries_count = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    tries_count += 1

                    if tries_count <= self._num_retries:
                        retry_in = tries_count * RETRY_SLEEP_TIME
                        self._log(
                            'Exception occurred: %s. Retry in %s seconds.' % (
                                exc, retry_in))

                        time.sleep(retry_in)
                    else:
                        raise

        return wrap_it

    @contextlib.contextmanager
    def retry_context(self, func):
        """Use retry_errors object as a context manager.

        :param func: decorated function.
        """

        yield self(func)


def get_port(port_or_port_re, process_name):
    """Given the regular expression, extract port from the process name.

    :param str port_or_port_re: whether integer port or port regular expression.
    :param str process_name: process name.

    :rtype: int|None
    """

    if isinstance(port_or_port_re, int):
        return port_or_port_re

    try:
        return int(port_or_port_re)
    except ValueError:
        pass

    match = re.match(port_or_port_re, process_name)

    if match:
        try:
            groups = match.groups()
            if len(groups) == 1:
                return int(groups[0])
        except (ValueError, TypeError) as err:
            raise errors.InvalidCheckConfig(err)

    raise errors.InvalidCheckConfig(
        'Could not extract port number for process name %s using regular '
        'expression %s' % (process_name, port_or_port_re))
