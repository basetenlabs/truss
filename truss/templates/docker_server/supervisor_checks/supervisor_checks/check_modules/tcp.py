"""Process check based on TCP connection status.
"""

import socket

from supervisor_checks import errors
from supervisor_checks import utils
from supervisor_checks.check_modules import base

__author__ = 'vovanec@gmail.com'


DEFAULT_RETRIES = 2
DEFAULT_TIMEOUT = 15

LOCALHOST = '127.0.0.1'


class TCPCheck(base.BaseCheck):
    """Process check based on TCP connection status.
    """

    NAME = 'tcp'

    def __call__(self, process_spec):

        timeout = self._config.get('timeout', DEFAULT_TIMEOUT)
        num_retries = self._config.get('num_retries', DEFAULT_RETRIES)

        try:
            port = utils.get_port(self._config['port'], process_spec['name'])
            with utils.retry_errors(num_retries, self._log).retry_context(
                    self._tcp_check) as retry_tcp_check:
                return retry_tcp_check(process_spec['name'], port, timeout)
        except errors.InvalidPortSpec:
            self._log('ERROR: Could not extract the HTTP port for process '
                      'name %s using port specification %s.',
                      process_spec['name'], self._config['port'])

            return True
        except Exception as exc:
            self._log('Check failed: %s', exc)

        return False

    def _tcp_check(self, process_name, port, timeout):

        self._log('Trying to connect to TCP port %s for process %s',
                  port, process_name)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((LOCALHOST, port))
        sock.close()

        self._log('Successfully connected to TCP port %s for process %s',
                  port, process_name)

        return True

    def _validate_config(self):

        if 'port' not in self._config:
            raise errors.InvalidCheckConfig(
                'Required `port` parameter is missing in %s check config.' % (
                    self.NAME,))
