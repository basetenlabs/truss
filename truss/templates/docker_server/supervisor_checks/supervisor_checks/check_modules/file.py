import time
import os
from supervisor_checks import errors
from supervisor_checks import utils
from supervisor_checks.check_modules import base


class FileCheck(base.BaseCheck):
    NAME = "file"

    def __call__(self, process_spec):
        notification_filepath = self._config["filepath"]
        if notification_filepath is None:
            notification_filepath = utils.NotificationFile.get_filepath(self._config["root_dir"], process_spec["group"], process_spec["name"], process_spec["pid"])

        try:
            stat = os.stat(notification_filepath, follow_symlinks=False)
            return time.time() - stat.st_ctime <= self._config["timeout"]
        except OSError:
            self._log("ERROR: Could not stat file: %s" % (notification_filepath,))
            return not self._config["fail_on_error"]


    def _validate_config(self):
        if "timeout" not in self._config:
            raise errors.InvalidCheckConfig(
                'Required `timeout` parameter is missing in %s check config.' % (
                    self.NAME,))
        
        if not isinstance(self._config['timeout'], int):
            raise errors.InvalidCheckConfig(
                '`timeout` parameter must be int type in %s check config.' % (
                    self.NAME,))

        if "fail_on_error" not in self._config:
            raise errors.InvalidCheckConfig(
                'Required `fail_on_error` parameter is missing in %s check config.' % (
                    self.NAME,))

        if "filepath" not in self._config:
            raise errors.InvalidCheckConfig(
                'Required `filepath` parameter is missing in %s check config.' % (
                    self.NAME,))

        if "root_dir" not in self._config:
            raise errors.InvalidCheckConfig(
                'Required `root_dir` parameter is missing in %s check config.' % (
                    self.NAME,))