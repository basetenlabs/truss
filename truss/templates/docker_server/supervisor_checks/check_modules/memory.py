"""Process check based on RSS memory usage.
"""

import psutil

from supervisor_checks import errors
from supervisor_checks.check_modules import base

__author__ = 'vovanec@gmail.com'


class MemoryCheck(base.BaseCheck):
    """Process check based on memory usage.
    """

    NAME = 'memory'

    def __call__(self, process_spec):

        pid = process_spec['pid']
        process_name = process_spec['name']

        if self._config.get('cumulative', False):
            rss = self._get_cumulative_rss(pid, process_name)
        else:
            rss = self._get_rss(pid, process_name)

        self._log('Total memory consumed by process %s is %s KB',
                  process_name, rss)

        if rss > self._config['max_rss']:
            self._log('Memory usage for process %s is above the configured '
                      'threshold: %s KB vs %s KB.', process_name,
                      rss, self._config['max_rss'])

            return False

        return True

    def _get_rss(self, pid, process_name):
        """Get RSS used by process.
        """

        self._log('Checking for RSS memory used by process %s', process_name)

        return int(psutil.Process(pid).memory_info().rss / 1024)

    def _get_cumulative_rss(self, pid, process_name):
        """Get cumulative RSS used by process and all its children.
        """

        self._log('Checking for cumulative RSS memory used by process %s',
                  process_name)

        parent = psutil.Process(pid)
        rss_total = parent.memory_info().rss
        for child_process in parent.children(recursive=True):
            rss_total += child_process.memory_info().rss

        return int(rss_total / 1024)

    def _validate_config(self):

        if 'max_rss' not in self._config:
            raise errors.InvalidCheckConfig(
                'Required `max_rss` parameter is missing in %s check config.'
                % (self.NAME,))

        if not isinstance(self._config['max_rss'], (int, float)):
            raise errors.InvalidCheckConfig(
                '`max_rss` parameter must be numeric type in %s check config.'
                % (self.NAME,))
