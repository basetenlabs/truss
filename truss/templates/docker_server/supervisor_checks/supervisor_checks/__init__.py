"""Framework to build health checks for Supervisor-based services.

Health check programs are supposed to run as event listeners in Supervisor
environment. Here's typical configuration example:

    [eventlistener:example_check]
    command=/usr/local/bin/supervisor_example_check
    stderr_logfile = /var/log/supervisor/supervisor_example_check-stderr.log
    stdout_logfile = /var/log/supervisor/supervisor_example_check-stdout.log
    events=TICK_60

While framework provides the set of ready-for-use health check classes(
tcp, http, xmlrpc, memory etc), it can be easily extended by adding custom
health checks. Here's really simple example of adding custom check:

    from supervisor_checks.check_modules import base
    from supervisor_checks import check_runner

    class ExampleCheck(base.BaseCheck):

        NAME = 'example'

        def __call__(self, process_spec):

            # Always return True

            return True

    check_runner.CheckRunner(
        'example_check', 'some_process_group', [(ExampleCheck, {})]).run()
"""

__author__ = 'vovanec@gmail.com'
