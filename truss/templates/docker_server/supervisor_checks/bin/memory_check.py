#! /usr/bin/env python3

"""Example configuration:

[eventlistener:example_check]
command=/usr/local/bin/supervisor_memory_check -n example_check -m 102400 -c -g example_service
events=TICK_60
"""

import argparse
import sys

from supervisor_checks import check_runner
from supervisor_checks.check_modules import memory


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
        '-m', '--msx-rss', dest='max_rss', type=int, required=True,
        help='Maximum memory allowed to use by process, KB.')
    parser.add_argument(
        '-c', '--cumulative', dest='cumulative', action='store_true',
        help='Recursively calculate memory used by all process children.')

    return parser


def main():

    arg_parser = _make_argument_parser()
    args = arg_parser.parse_args()

    checks_config = [(memory.MemoryCheck, {'max_rss': args.max_rss,
                                           'cumulative': args.cumulative})]

    return check_runner.CheckRunner(
        args.check_name, args.process_group, args.process_name, checks_config).run()


if __name__ == '__main__':

    sys.exit(main())
