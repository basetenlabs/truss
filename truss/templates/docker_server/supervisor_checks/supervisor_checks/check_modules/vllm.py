"""Process check based on HTTP query.
"""

import base64
import json

from supervisor_checks import errors
from supervisor_checks import utils
from supervisor_checks.check_modules import base
from supervisor_checks.compat import httplib

__author__ = 'vovanec@gmail.com'


DEFAULT_RETRIES = 2
DEFAULT_TIMEOUT = 15

LOCALHOST = '127.0.0.1'


class VLLMCheck(base.BaseCheck):
    """Process check based on HTTP query.
    """

    HEADERS = {'User-Agent': 'vllm_check'}
    NAME = 'vllm'

    def __call__(self, process_spec):

        try:
            port = utils.get_port(self._config['port'], process_spec['name'])
            try:
                readiness_pass = self._http_check(process_spec['name'], port, self._config['readiness_url'])
            except Exception as exc:
                self._log('Readiness check failed: %s', exc)
                readiness_pass = False
                
            try:
                liveness_pass = self._http_check(process_spec['name'], port, self._config['liveness_url'])
            except Exception as exc:
                self._log('Liveness check failed: %s', exc)
                liveness_pass = False

            if readiness_pass:
                return liveness_pass
            else:
                self._log('Readiness check failed, skip health check for process %s', process_spec['name'])
                return True
            
        except errors.InvalidPortSpec:
            self._log('ERROR: Could not extract the HTTP port for process '
                      'name %s using port specification %s.',
                      process_spec['name'], self._config['port'])
        except Exception as exc:
            self._log('Check failed: %s', exc)

        return True

    def _http_check(self, process_name, port, url):

        self._log('Querying URL http://%s:%s%s for process %s',
                  LOCALHOST, port, url,
                  process_name)

        host_port = '%s:%s' % (LOCALHOST, port,)
        num_retries = self._config.get('num_retries', DEFAULT_RETRIES)
        timeout = self._config.get('timeout', DEFAULT_TIMEOUT)
        username = self._config.get('username')
        password = self._config.get('password')

        with utils.retry_errors(num_retries, self._log).retry_context(
                self._make_http_request) as retry_http_request:
            res = retry_http_request(
                host_port, url, timeout, username=username, password=password)

        self._log('Status contacting URL http://%s%s for process %s: '
                  '%s %s' % (host_port, url, process_name,
                             res.status, res.reason))

        if res.status != httplib.OK:
            raise httplib.HTTPException(
                'Bad HTTP status code: %s' % (res.status,))

        return True

    def _make_http_request(self, host_port, url, timeout,
                           username=None, password=None):

        connection = httplib.HTTPConnection(host_port, timeout=timeout)
        headers = self.HEADERS.copy()

        if username and password:
            auth_str = '%s:%s' % (username, password)
            headers['Authorization'] = 'Basic %s' % base64.b64encode(
                auth_str.encode()).decode()

        config_headers = self._config.get('headers')
        if config_headers:
            headers.update(config_headers)
            # auto apply content type if json argument is passed in
            if self._config.get('json'):
                headers['Content-Type'] = 'application/json'

        body = self._config.get('body')
        json_body = self._config.get('json')
        if json_body:
            body = json.dumps(json_body)

        connection.request(
            self._config.get('method'), url, body,
            headers=headers)

        return connection.getresponse()

    def _validate_config(self):

        if 'readiness_url' not in self._config:
            raise errors.InvalidCheckConfig(
                'Required `readiness_url` parameter is missing in %s check config.' % (
                    self.NAME,))

        if not isinstance(self._config['readiness_url'], str):
            raise errors.InvalidCheckConfig(
                '`readiness_url` parameter must be string type in %s check config.' % (
                    self.NAME,))
        
        if 'liveness_url' not in self._config:
            raise errors.InvalidCheckConfig(
                'Required `liveness_url` parameter is missing in %s check config.' % (
                    self.NAME,))
        
        if not isinstance(self._config['liveness_url'], str):
            raise errors.InvalidCheckConfig(
                '`liveness_url` parameter must be string type in %s check config.' % (
                    self.NAME,))

        if 'port' not in self._config:
            raise errors.InvalidCheckConfig(
                'Required `port` parameter is missing in %s check config.' % (
                    self.NAME,))
