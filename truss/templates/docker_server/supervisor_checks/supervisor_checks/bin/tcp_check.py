"""Health check based on TCP connection status.

Example configuration:

[eventlistener:example_check]
command=/usr/local/bin/supervisor_tcp_check -n example_service_check -u /ping -t 30 -r 3 -g example_service -p 8080
events=TICK_60
"""

import argparse
import sys

from supervisor_checks import check_runner
from supervisor_checks.check_modules import tcp


__author__ = 'vovanec@gmail.net'


def _make_argument_parser():
    """Create the option parser.
    """

    parser = argparse.ArgumentParser(
        description='Run TCP check program.')
    parser.add_argument('-n', '--check-name', dest='check_name',
                        type=str, required=True, default=None,
                        help='Check name.')
    parser.add_argument('-g', '--process-group', dest='process_group',
                        type=str, default=None,
                        help='Supervisor process group name.')
    parser.add_argument('-N', '--process-name', dest='process_name',
                        type=str, default=None,
                        help='Supervisor process name. Process group argument is ignored if this ' +
                             'is passed in')
    parser.add_argument(
        '-p', '--port', dest='port', type=str,
        default=None, required=True,
        help='TCP port to query. Can be integer or regular expression which '
             'will be used to extract port from a process name.')
    parser.add_argument(
        '-t', '--timeout', dest='timeout', type=int, required=False,
        default=tcp.DEFAULT_TIMEOUT,
        help='Connection timeout. Default: %s' % (tcp.DEFAULT_TIMEOUT,))
    parser.add_argument(
        '-r', '--num-retries', dest='num_retries', type=int,
        default=tcp.DEFAULT_RETRIES, required=False,
        help='Connection retries. Default: %s' % (tcp.DEFAULT_RETRIES,))

    return parser


def main():

    arg_parser = _make_argument_parser()
    args = arg_parser.parse_args()

    checks_config = [(tcp.TCPCheck, {'timeout': args.timeout,
                                     'num_retries': args.num_retries,
                                     'port': args.port})]

    return check_runner.CheckRunner(
        args.check_name, args.process_group, args.process_name, checks_config).run()


if __name__ == '__main__':

    sys.exit(main())
