#! /usr/bin/env python3

"""Example configuration:

[eventlistener:example_check]
command=/usr/local/bin/supervisor_xmlrpc_check -g example_service -n example_check -u /ping -r 3 -p 8080
events=TICK_60
"""

import argparse
import sys

from supervisor_checks import check_runner
from supervisor_checks.check_modules import xmlrpc

__author__ = 'vovanec@gmail.com'


def _make_argument_parser():
    """Create the option parser.
    """

    parser = argparse.ArgumentParser(
        description='Run XML RPC check program.')
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
    parser.add_argument('-u', '--url', dest='url', type=str,
                        help='XML RPC check url', required=False, default=None)
    parser.add_argument('-s', '--socket-path', dest='sock_path', type=str,
                        help='Full path to XML RPC server local socket',
                        required=False, default=None)
    parser.add_argument('-S', '--socket-dir', dest='sock_dir', type=str,
                        help='Path to XML RPC server socket directory. Socket '
                             'name will be constructed using process name: '
                             '<process_name>.sock.',
                        required=False, default=None)
    parser.add_argument('-m', '--method', dest='method', type=str,
                        help='XML RPC method name. Default is %s' % (
                            xmlrpc.DEFAULT_METHOD,), required=False,
                        default=xmlrpc.DEFAULT_METHOD)
    parser.add_argument('-U', '--username', dest='username', type=str,
                        help='XMLRPC check username', required=False,
                        default=None)
    parser.add_argument('-P', '--password', dest='password', type=str,
                        help='XMLRPC check password', required=False,
                        default=None)
    parser.add_argument(
        '-p', '--port', dest='port', type=str,
        default=None, required=False,
        help='Port to query. Can be integer or regular expression which '
             'will be used to extract port from a process name.')
    parser.add_argument(
        '-r', '--num-retries', dest='num_retries', type=int,
        default=xmlrpc.DEFAULT_RETRIES, required=False,
        help='Connection retries. Default: %s' % (xmlrpc.DEFAULT_RETRIES,))

    return parser


def main():

    arg_parser = _make_argument_parser()
    args = arg_parser.parse_args()

    checks_config = [(xmlrpc.XMLRPCCheck, {'url': args.url,
                                           'sock_path': args.sock_path,
                                           'sock_dir': args.sock_dir,
                                           'num_retries': args.num_retries,
                                           'port': args.port,
                                           'method': args.method,
                                           'username': args.username,
                                           'password': args.password,
                                           })]

    return check_runner.CheckRunner(
        args.check_name, args.process_group, args.process_name, checks_config).run()


if __name__ == '__main__':

    sys.exit(main())

