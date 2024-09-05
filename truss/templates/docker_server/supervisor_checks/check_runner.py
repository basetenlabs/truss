"""Instance of CheckRunner runs the set of pre-configured checks
against the process running under SupervisorD.
"""

import concurrent.futures
import datetime
import os
import select
import signal
import sys
import threading

from supervisor import childutils
from supervisor.options import make_namespec, split_namespec
from supervisor.states import ProcessStates

from supervisor_checks.compat import xmlrpclib

__author__ = 'vovanec@gmail.com'


# Process spec keys
STATE_KEY = 'state'
NAME_KEY = 'name'
GROUP_KEY = 'group'
EVENT_NAME_KEY = 'eventname'

MAX_THREADS = 16
TICK_EVENTS = set(['TICK_5', 'TICK_60', 'TICK_3600'])


class AboutToShutdown(Exception):
    """Raised from supervisor events read loop, when
    application is about to shutdown.
    """

    pass


class CheckRunner(object):
    """SupervisorD checks runner.
    """

    def __init__(self, check_name, process_group, process_name, checks_config, env=None):
        """Constructor.

        :param str check_name: the name of check to display in log.
        :param str process_group: the name of the process group.
        :param list checks_config: the list of check module configurations
               in format [(check_class, check_configuration_dictionary)]
        :param dict env: environment.
        """

        self._environment = env or os.environ
        self._name = check_name
        self._checks_config = checks_config
        self._checks = self._init_checks()
        self._process_group = process_group
        # represents specific process name
        self._process_name = process_name
        self._group_check_name = '%s_check' % (self._process_display_name(),)
        self._rpc_client = childutils.getRPCInterface(self._environment)
        self._stop_event = threading.Event()

    def run(self):
        """Run main check loop.
        """

        self._log('Starting the health check for %s process '
                  'Checks config: %s', self._process_display_name(), self._checks_config)

        self._install_signal_handlers()

        while not self._stop_event.is_set():

            try:
                event_type = self._wait_for_supervisor_event()
            except AboutToShutdown:
                self._log(
                    'Health check for %s process has been told to stop.',
                    self._process_display_name())

                break

            if event_type in TICK_EVENTS:
                self._check_processes()
            else:
                self._log('Received unsupported event type: %s', event_type)

            childutils.listener.ok(sys.stdout)

        self._log('Done.')

    def _check_processes(self):
        """Run single check loop for process group or name.
        """

        process_specs = self._get_process_spec_list(ProcessStates.RUNNING)
        if process_specs:
            if len(process_specs) == 1:
                self._check_and_restart(process_specs[0])
            else:
                # Query and restart in multiple threads simultaneously.
                with concurrent.futures.ThreadPoolExecutor(MAX_THREADS) as pool:
                    for process_spec in process_specs:
                        pool.submit(self._check_and_restart, process_spec)
        else:
            self._log(
                'No processes in state RUNNING found for process %s',
                self._process_display_name())

    def _check_and_restart(self, process_spec):
        """Run checks for the process and restart if needed.
        """

        for check in self._checks:
            self._log('Performing `%s` check for process name %s',
                      check.NAME, process_spec['name'])

            try:
                if not check(process_spec):
                    self._log('`%s` check failed for process %s. Trying to '
                              'restart.', check.NAME, process_spec['name'])

                    return self._restart_process(process_spec)
                else:
                    self._log('`%s` check succeeded for process %s',
                              check.NAME, process_spec['name'])
            except Exception as exc:
                self._log('`%s` check raised error for process %s: %s',
                          check.NAME, process_spec['name'], exc)

    def _init_checks(self):
        """Init check instances.

        :rtype: list
        """

        checks = []
        for check_class, check_cfg in self._checks_config:
            checks.append(check_class(check_cfg, self._log))

        return checks

    def _get_process_spec_list(self, state=None):
        """Get the list of processes in a process group or name.

        If process_name doesn't exist then get all processes in the defined group
        If process_name exists then get only the process(es) that match that name
        """

        process_specs = []
        for process_spec in self._rpc_client.supervisor.getAllProcessInfo():
            if not self._process_name:
                if (process_spec[GROUP_KEY] == self._process_group and
                    (state is None or process_spec[STATE_KEY] == state)):
                    process_specs.append(process_spec)
            else:
                if ((process_spec[GROUP_KEY], process_spec[NAME_KEY]) == 
                    split_namespec(self._process_name) and
                    (state is None or process_spec[STATE_KEY] == state)):
                    process_specs.append(process_spec)

        return process_specs

    def _restart_process(self, process_spec):
        """Restart a process.
        """

        if not self._process_name:
            name_spec = make_namespec(
                process_spec[GROUP_KEY], process_spec[NAME_KEY])
        else:
            name_spec_tuple = split_namespec(self._process_name)
            name_spec = make_namespec(name_spec_tuple[0], name_spec_tuple[1])

        rpc_client = childutils.getRPCInterface(self._environment)

        process_spec = rpc_client.supervisor.getProcessInfo(name_spec)
        if process_spec[STATE_KEY] is ProcessStates.RUNNING:
            self._log('Trying to stop process %s', name_spec)

            try:
                rpc_client.supervisor.stopProcess(name_spec)
                self._log('Stopped process %s', name_spec)
            except xmlrpclib.Fault as exc:
                self._log('Failed to stop process %s: %s', name_spec, exc)

            try:
                self._log('Starting process %s', name_spec)
                rpc_client.supervisor.startProcess(name_spec, False)
            except xmlrpclib.Fault as exc:
                self._log('Failed to start process %s: %s', name_spec, exc)

        else:
            self._log('%s not in RUNNING state, cannot restart', name_spec)

    def _log(self, msg, *args):
        """Write message to STDERR.

        :param str msg: string message.
        """

        curr_dt = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')

        sys.stderr.write(
            '%s [%s] %s\n' % (curr_dt, self._name, msg % args,))

        sys.stderr.flush()

    def _install_signal_handlers(self):
        """Install signal handlers.
        """

        self._log('Installing signal handlers.')

        for sig in (signal.SIGINT, signal.SIGUSR1, signal.SIGHUP,
                    signal.SIGTERM, signal.SIGQUIT):
            signal.signal(sig, self._signal_handler)

    def _signal_handler(self, signum, _):
        """Signal handler.
        """

        self._log('Got signal %s', signum)

        self._stop_event.set()

    def _wait_for_supervisor_event(self):
        """Wait for supervisor events.
        """

        childutils.listener.ready(sys.stdout)

        while not self._stop_event.is_set():
            try:
                rdfs, _, _ = select.select([sys.stdin], [], [], .5)
            except InterruptedError:
                continue

            if rdfs:
                headers = childutils.get_headers(rdfs[0].readline())
                # Read the payload to make read buffer empty.
                _ = sys.stdin.read(int(headers['len']))
                event_type = headers[EVENT_NAME_KEY]
                self._log('Received %s event from supervisor', event_type)

                return event_type

        raise AboutToShutdown

    def _process_display_name(self):
        return self._process_name or self._process_group
