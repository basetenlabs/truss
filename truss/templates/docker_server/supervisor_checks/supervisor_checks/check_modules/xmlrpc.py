"""Process check based on call to XML RPC server.
"""

import supervisor.xmlrpc

from supervisor_checks import errors
from supervisor_checks import utils
from supervisor_checks.check_modules import base
from supervisor_checks.compat import xmlrpclib

__author__ = 'vovanec@gmail.com'


DEFAULT_RETRIES = 2
DEFAULT_METHOD = 'status'

LOCALHOST = '127.0.0.1'


class XMLRPCCheck(base.BaseCheck):
    """Process check based on query to XML RPC server.
    """

    NAME = 'xmlrpc'

    def __call__(self, process_spec):

        try:
            process_name = process_spec['name']
            retries_left = self._config.get('num_retries', DEFAULT_RETRIES)
            method_name = self._config.get('method', DEFAULT_METHOD)
            username = self._config.get('username')
            password = self._config.get('password')

            server_url = self._get_server_url(process_name)
            if not server_url:
                return True

            self._log('Querying XML RPC server at %s, method %s for process %s',
                      server_url, method_name, process_name)

            with utils.retry_errors(retries_left, self._log).retry_context(
                    self._xmlrpc_check) as retry_xmlrpc_check:

                return retry_xmlrpc_check(process_name, server_url, method_name,
                                          username=username, password=password)
        except Exception as exc:
            self._log('Check failed: %s', exc)

        return False

    def _xmlrpc_check(self, process_name, server_url, method_name,
                      username=None, password=None):

        try:
            xmlrpc_result = getattr(
                self._get_rpc_client(server_url,
                                     username=username,
                                     password=password), method_name)()

            self._log('Successfully contacted XML RPC server at %s, '
                      'method %s for process %s. Result: %s', server_url,
                      method_name, process_name, xmlrpc_result)

            return True
        except xmlrpclib.Fault as err:
            self._log('XML RPC server returned error: %s', err)

        return False

    def _validate_config(self):

        one_of_required = set(['url', 'sock_path', 'sock_dir'])

        param_intersection = one_of_required.intersection(self._config)
        if not param_intersection:
            raise errors.InvalidCheckConfig(
                'One of required parameters: `url`, `sock_path` or `sock_dir` '
                'is missing in %s check config.' % (self.NAME,))

        if len(param_intersection) > 1:
            raise errors.InvalidCheckConfig(
                '`url`, `sock_path` and `sock_dir` must be mutually exclusive'
                'in %s check config.' % (self.NAME,))

        if 'url' in self._config and 'port' not in self._config:
            raise errors.InvalidCheckConfig(
                'When `url` parameter is specified, `port` parameter is '
                'required in %s check config.' % (self.NAME,))

    def _get_server_url(self, process_name):
        """Construct XML RPC server URL.

        :param str process_name: process name.
        :rtype: str|None
        """

        url = self._config.get('url')

        if url:
            try:
                port = utils.get_port(self._config['port'],
                                      process_name)

                return 'http://%s:%s%s' % (LOCALHOST, port, url)
            except errors.InvalidPortSpec:
                self._log('ERROR: Could not extract the HTTP port for '
                          'process name %s using port specification %s.',
                          process_name, self._config['port'])
        else:
            sock_path = self._config.get('sock_path')

            if not sock_path:
                sock_dir = self._config.get('sock_dir')

                if not sock_dir:
                    self._log('ERROR: Could not construct XML RPC socket '
                              'path using configuration provided. sock_dir '
                              'or sock_path argument must be specified.')
                    return None

                sock_path = 'unix://%s/%s.sock' % (sock_dir, process_name,)

            if not sock_path.startswith('unix://'):
                sock_path = 'unix://%s' % (sock_path,)

            return sock_path

    @staticmethod
    def _get_rpc_client(server_url, username=None, password=None):

        return xmlrpclib.ServerProxy(
            'http://127.0.0.1', supervisor.xmlrpc.SupervisorTransport(
                username, password, server_url))
