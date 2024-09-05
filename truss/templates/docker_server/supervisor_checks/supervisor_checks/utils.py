"""Utility functions.
"""


import contextlib
import functools
import re
import time
import os
import tempfile

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


class _TemporaryFileWrapper:
    """Temporary file wrapper

    This class provides a wrapper around files opened for
    temporary use.  In particular, it seeks to automatically
    remove the file when it is no longer needed.
    """

    def __init__(self, file, name, delete=True):
        self.file = file
        self.name = name
        self.delete = delete
        self.close_called = False

    def __getattr__(self, name):
        return getattr(self.file, name)

    def close(self, unlink=os.unlink):
        if not self.close_called and self.file is not None:
            self.close_called = True
            try:
                self.file.close()
            finally:
                if self.delete:
                    unlink(self.name)

    def __del__(self):
        self.close()


_open_flags = os.O_CREAT
if hasattr(os, "O_NOFOLLOW"):
    _open_flags |= os.O_NOFOLLOW

class NotificationFile:
    @staticmethod
    def get_filename(process_group, process_name, pid):
        return f"{process_group!s}-{process_name!s}-{pid!s}"

    @staticmethod
    def get_filepath(root_dir=None, process_group=None, process_name=None, pid=None):
        root_dir = root_dir if root_dir is not None else tempfile.gettempdir()
        process_group = process_group if process_group is not None else os.getenv("SUPERVISOR_GROUP_NAME")
        process_name = process_name if process_name is not None else os.getenv("SUPERVISOR_PROCESS_NAME")
        pid = pid if pid is not None else os.getpid()
        return os.path.join(root_dir, NotificationFile.get_filename(process_group, process_name, pid))

    def __init__(self, filepath=None, root_dir=None, delete=True):
        """
        Creates a NotificationFile object used to indicate a heartbeat.
        Only supports UNIX.

        param str filepath: optional filepath to use as notification file
        param str root_dir: optional root_dir to use for the notification file (default: tempfile.gettempdir())
        param bool delete: wether to delete the notification file after fd is closed 
        """
        if filepath is None:
            filepath = self.get_filepath(root_dir=root_dir)

        def opener(file, flags):
            flags |= _open_flags
            return os.open(file, flags, mode=0o000)

        fd = open(filepath, "rb", buffering=0, opener=opener)
        self._tmp = _TemporaryFileWrapper(fd, filepath, delete=delete)

        self.spinner = 0

    def notify(self):
        self.spinner = (self.spinner + 1) % 2
        os.fchmod(self._tmp.fileno(), self.spinner)

    def close(self):
        return self._tmp.close()
