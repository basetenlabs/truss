#! /usr/bin/env python3

"""Example configuration:

[eventlistener:example_check]
command=/usr/local/bin/supervisor_http_check -g example_service -n example_check -u /ping -t 30 -r 3 -p 8080
events=TICK_60
"""

import argparse
import json
import sys

from supervisor_checks import check_runner
from supervisor_checks.check_modules import vllm

__author__ = 'vovanec@gmail.com'


def _make_argument_parser():
    """Create the option parser.
    """

    parser = argparse.ArgumentParser(
        description='Run HTTP check program.')
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
    parser.add_argument('-R', '--readiness_url', dest='readiness_url', type=str,
                        help='VLLM check readiness_url', required=True, default=None)
    parser.add_argument('-L', '--liveness_url', dest='liveness_url', type=str,
                        help='VLLM check liveness_url', required=True, default=None)
    parser.add_argument('-m', '--method', dest='method', type=str,
                        help='HTTP request method (GET, POST, PUT...)', default='GET')
    parser.add_argument('-j', '--json', dest='json', type=json.loads,
                        help='HTTP json body, auto sets content-type header to application/json',
                        default=None)
    parser.add_argument('-b', '--body', dest='body', type=str,
                        help='HTTP body, will be ignored if json body pass in', default=None)
    parser.add_argument('-H', '--headers', dest='headers', type=json.loads,
                        help='HTTP headers as json', default=None)
    parser.add_argument('-U', '--username', dest='username', type=str,
                        help='HTTP check username', required=False,
                        default=None)
    parser.add_argument('-P', '--password', dest='password', type=str,
                        help='HTTP check password', required=False,
                        default=None)
    parser.add_argument(
        '-p', '--port', dest='port', type=str,
        default=None, required=True,
        help='HTTP port to query. Can be integer or regular expression which '
             'will be used to extract port from a process name.')
    parser.add_argument(
        '-t', '--timeout', dest='timeout', type=int, required=False,
        default=vllm.DEFAULT_TIMEOUT,
        help='Connection timeout. Default: %s' % (vllm.DEFAULT_TIMEOUT,))
    parser.add_argument(
        '-r', '--num-retries', dest='num_retries', type=int,
        default=vllm.DEFAULT_RETRIES, required=False,
        help='Connection retries. Default: %s' % (vllm.DEFAULT_RETRIES,))

    return parser


def main():

    arg_parser = _make_argument_parser()
    args = arg_parser.parse_args()

    checks_config = [(vllm.VLLMCheck, {'readiness_url': args.readiness_url,
                                        'liveness_url': args.liveness_url,
                                       'timeout': args.timeout,
                                       'num_retries': args.num_retries,
                                       'method': args.method,
                                       'json': args.json,
                                       'body': args.body,
                                       'headers': args.headers,
                                       'port': args.port,
                                       'username': args.username,
                                       'password': args.password,
                                       })]
    return check_runner.CheckRunner(
        args.check_name, args.process_group, args.process_name, checks_config).run()


if __name__ == '__main__':

    sys.exit(main())
