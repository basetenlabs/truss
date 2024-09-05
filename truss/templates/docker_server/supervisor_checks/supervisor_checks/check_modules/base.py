"""Base class for checks.
"""

__author__ = 'vovanec@gmail.com'


class BaseCheck(object):
    """Base class for checks.
    """

    NAME = None

    def __init__(self, check_config, log):
        """Constructor.

        :param dict check_config: implementation specific check config.
        :param (str) -> None log: logging function.
        """

        self._config = check_config
        self._validate_config()
        self.__log = log

    def __call__(self, process_spec):
        """Run single check.

        :param dict process_spec: process specification dictionary as returned
               by SupervisorD API.

        :return: True is check succeeded, otherwise False. If check failed -
                 monitored process will be automatically restarted.

        :rtype: bool
        """

        raise NotImplementedError

    def _validate_config(self):
        """Method may be implemented in subclasses. Should return None or
        raise InvalidCheckConfig in case if configuration is invalid.

        Here's typical example of parameter check:

          if 'url' not in self._config:
              raise errors.InvalidCheckConfig(
                  'Required `url` parameter is missing in %s check config.' % (
                      self.NAME,))
        """

        pass

    def _log(self, msg, *args):
        """Log check message.

        :param str msg: log message.
        """

        self.__log('%s: %s' % (self.__class__.__name__, msg % args))
