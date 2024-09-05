#! /usr/bin/env python3

"""Example configuration(restart process when it consumes more than
100% CPU within 30 minutes):

[eventlistener:example_check]
command=/usr/local/bin/supervisor_cpu_check -n example_check -p 100 -i 1800 -g example_service
events=TICK_60
"""

import argparse
import sys

from supervisor_checks import check_runner
from supervisor_checks.check_modules import cpu


__author__ = 'vovanec@gmail.com'


def _make_argument_parser():
    """Create the option parser.
    """

    parser = argparse.ArgumentParser(
        description='Run memory check program.')
    parser.add_argument('-n', '--check-name', dest='check_name',
                        type=str, required=True, default=None,
                        help='Health check name.')
    parser.add_argument('-g', '--process-group', dest='process_group',
                        type=str, default=None,
                        help='Supervisor process group name.')
    parser.add_argument('-N', '--process-name', dest='process_name',
                        type=str, default=None,
                        help='Supervisor process name. Process group argument is ignored if this ' +
                             'is passed in')
    parser.add_argument(
        '-p', '--max-cpu-percent', dest='max_cpu', type=int, required=True,
        help='Maximum CPU percent usage allowed to use by process '
             'within time interval.')

    parser.add_argument(
        '-i', '--interval', dest='interval', type=int, required=True,
        help='How long process is allowed to use CPU over threshold, seconds.')

    return parser


def main():

    arg_parser = _make_argument_parser()
    args = arg_parser.parse_args()

    checks_config = [(cpu.CPUCheck, {'max_cpu': args.max_cpu,
                                     'interval': args.interval})]

    return check_runner.CheckRunner(
        args.check_name, args.process_group, args.process_name, checks_config).run()


if __name__ == '__main__':

    sys.exit(main())
