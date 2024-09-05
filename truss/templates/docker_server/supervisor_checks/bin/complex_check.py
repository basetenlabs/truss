#! /usr/bin/env python3

"""Example configuration:

[eventlistener:example_check]
command=/usr/local/bin/supervisor_complex_check -n example_check -g example_service -c '{"memory":{"cumulative":true,"max_rss":4194304},"http":{"timeout":15,"port":8090,"url":"\/ping","num_retries":3}}'
events=TICK_60
"""


import argparse
import json
import sys

from supervisor_checks import check_runner
from supervisor_checks.check_modules import cpu
from supervisor_checks.check_modules import http
from supervisor_checks.check_modules import memory
from supervisor_checks.check_modules import tcp
from supervisor_checks.check_modules import xmlrpc

__author__ = 'vovanec@gmail.com'


CHECK_CLASSES = {http.HTTPCheck.NAME: http.HTTPCheck,
                 memory.MemoryCheck.NAME: memory.MemoryCheck,
                 tcp.TCPCheck.NAME: tcp.TCPCheck,
                 xmlrpc.XMLRPCCheck.NAME: xmlrpc.XMLRPCCheck,
                 cpu.CPUCheck.NAME: cpu.CPUCheck}


def _make_argument_parser():
    """Create the option parser.
    """

    parser = argparse.ArgumentParser(
        description='Run SupervisorD check program.')
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
    parser.add_argument('-c', '--check-config', dest='check_config', type=str,
                        help='Check config JSON', required=True, default=None)

    return parser


def main():

    arg_parser = _make_argument_parser()
    args = arg_parser.parse_args()

    checks_config_dict = json.loads(args.check_config)
    if not isinstance(checks_config_dict, dict):
        raise ValueError('Check config must be dictionary type!')

    checks_config = []
    for check_name, check_config in checks_config_dict.items():
        checks_config.append((CHECK_CLASSES[check_name], check_config))

    return check_runner.CheckRunner(
        args.check_name, args.process_group, args.process_name, checks_config).run()


if __name__ == '__main__':

    sys.exit(main())
