# Supervisor Health Checks

Framework to build health checks for Supervisor-based services.

Health check programs are supposed to run as event listeners in [Supervisor](http://supervisord.org)
environment. On check failure Supervisor will attempt to restart monitored
process.

Here's typical configuration example:

    [eventlistener:example_check]
    command=python <path_to_supervisor_check_program>
    stderr_logfile = /var/log/supervisor/supervisor_example_check-stderr.log
    stdout_logfile = /var/log/supervisor/supervisor_example_check-stdout.log
    events=TICK_60

Here's the list of check programs package provides out-of-box:

* _supervisor_http_check_: process check based on HTTP query.
* _supervisor_tcp_check_: process check based on TCP connection status.
* _supervisor_xmlrpc_check_: process check based on call to XML-RPC server.
* _supervisor_memory_check_: process check based on amount of memory consumed by process.
* _supervisor_cpu_check_: process check based on CPU percent usage within time interval.
* _supervisor_file_check_: process check based on file update timeout. (Only UNIX)
* _supervisor_complex_check_: complex check (run multiple checks at once).

For now, it is developed and supposed to work primarily with Python 3 and
Supervisor 4 branch. There's nominal Python 2.x support but it's not tested.

# Installation

Install and update using [pip](https://pypi.org/project/pip/)

    pip install supervisor_checks

## Developing Custom Check Modules

While framework provides the good set of ready-for-use health check classes,
it can be easily extended by adding application-specific custom health checks.

To implement custom check class, _check_modules.base.BaseCheck_ class must
be inherited:

```python
    class BaseCheck(object):
        """Base class for checks.
        """

        NAME = None

        def __call__(self, process_spec):
            """Run single check.

            :param dict process_spec: process specification dictionary as returned
                   by SupervisorD API.

            :return: True is check succeeded, otherwise False. If check failed -
                     monitored process will be automatically restarted.

            :rtype: bool
            """

        def _validate_config(self):
            """Method may be implemented in subclasses. Should return None or
            raise InvalidCheckConfig in case if configuration is invalid.

            Here's typical example of parameter check:

              if 'url' not in self._config:
                  raise errors.InvalidCheckConfig(
                      'Required `url` parameter is missing in %s check config.' % (
                          self.NAME,))
            """
```

Here's the example of adding custom check:

```python
    from supervisor_checks.check_modules import base
    from supervisor_checks import check_runner

    class ExampleCheck(base.BaseCheck):

        NAME = 'example'

        def __call__(self, process_spec):

            # Always return True
            return True

    if __name__ == '__main__':

        check_runner.CheckRunner(
            'example_check', 'some_process_group', [(ExampleCheck, {})]).run()
```

## Out-of-box checks

### HTTP Check

Process check based on HTTP query.

#### CLI

    $ /usr/local/bin/supervisor_http_check -h
    usage: supervisor_http_check [-h] -n CHECK_NAME -g PROCESS_GROUP -u URL -p
                                 PORT [-t TIMEOUT] [-r NUM_RETRIES]

    Run HTTP check program.

    optional arguments:
      -h, --help            show this help message and exit
      -n CHECK_NAME, --check-name CHECK_NAME
                            Health check name.
      -g PROCESS_GROUP, --process-group PROCESS_GROUP
                            Supervisor process group name.
      -N PROCESS_NAME, --process-name PROCESS_NAME
                            Supervisor process name. Process group argument is
                            ignored if this is passed in
      -u URL, --url URL     HTTP check url
      -m METHOD, --method METHOD
                            HTTP request method (GET, POST, PUT...)
      -j JSON, --json JSON  HTTP json body, auto sets content-type header to
                            application/json
      -b BODY, --body BODY  HTTP body, will be ignored if json body pass in
      -H HEADERS, --headers HEADERS
                            HTTP headers as json
      -U USERNAME, --username USERNAME
                            HTTP check username
      -P PASSWORD, --password PASSWORD
                            HTTP check password
      -p PORT, --port PORT  HTTP port to query. Can be integer or regular
                            expression which will be used to extract port from a
                            process name.
      -t TIMEOUT, --timeout TIMEOUT
                            Connection timeout. Default: 15
      -r NUM_RETRIES, --num-retries NUM_RETRIES
                            Connection retries. Default: 2

#### Configuration Examples

Query process running on port 8080 using URL _/ping_:

    [eventlistener:example_check]
    command=/usr/local/bin/supervisor_http_check -g example_service -n example_check -u /ping -t 30 -r 3 -p 8080
    events=TICK_60

Query process group using URL /ping. Each process is listening on it's own port.
Each process name is formed as _some-process-name\_port_ so particular port number can
be extracted using regular expression:

    [eventlistener:example_check]
    command=/usr/local/bin/supervisor_http_check -g example_service -n example_check -u /ping -t 30 -r 3 -p ".+_(\\d+)"
    events=TICK_60


### TCP Check

Process check based on TCP connection status.

#### CLI

    $ /usr/local/bin/supervisor_tcp_check -h
    usage: supervisor_tcp_check [-h] -n CHECK_NAME -g PROCESS_GROUP -p PORT
                                [-t TIMEOUT] [-r NUM_RETRIES]

    Run TCP check program.

    optional arguments:
      -h, --help            show this help message and exit
      -n CHECK_NAME, --check-name CHECK_NAME
                            Check name.
      -g PROCESS_GROUP, --process-group PROCESS_GROUP
                            Supervisor process group name.
      -N PROCESS_NAME, --process-name PROCESS_NAME
                            Supervisor process name. Process group argument is
                            ignored if this is passed in
      -p PORT, --port PORT  TCP port to query. Can be integer or regular
                            expression which will be used to extract port from a
                            process name.
      -t TIMEOUT, --timeout TIMEOUT
                            Connection timeout. Default: 15
      -r NUM_RETRIES, --num-retries NUM_RETRIES
                            Connection retries. Default: 2

#### Configuration Examples

Connect to process running on port 8080:

    [eventlistener:example_check]
    command=/usr/local/bin/supervisor_tcp_check -g example_service -n example_check -t 30 -r 3 -p 8080
    events=TICK_60

Query process group when each process is listening on it's own port.
Each process name is formed as _some-process-name\_port_ so particular port number can
be extracted using regular expression:

    [eventlistener:example_check]
    command=/usr/local/bin/supervisor_tcp_check -g example_service -n example_check -t 30 -r 3 -p ".+_(\\d+)"
    events=TICK_60


### XML-RPC Check

Process check based on call to XML-RPC server.

#### CLI

    $ /usr/local/bin/supervisor_xmlrpc_check -h
    usage: supervisor_xmlrpc_check [-h] -n CHECK_NAME -g PROCESS_GROUP [-u URL]
                                   [-s SOCK_PATH] [-S SOCK_DIR] [-p PORT]
                                   [-r NUM_RETRIES]

    Run XML RPC check program.

    optional arguments:
      -h, --help            show this help message and exit
      -n CHECK_NAME, --check-name CHECK_NAME
                            Health check name.
      -g PROCESS_GROUP, --process-group PROCESS_GROUP
                            Supervisor process group name.
      -N PROCESS_NAME, --process-name PROCESS_NAME
                            Supervisor process name. Process group argument is
                            ignored if this is passed in
      -u URL, --url URL     XML RPC check url
      -s SOCK_PATH, --socket-path SOCK_PATH
                            Full path to XML RPC server local socket
      -S SOCK_DIR, --socket-dir SOCK_DIR
                            Path to XML RPC server socket directory. Socket name
                            will be constructed using process name:
                            <process_name>.sock.
      -m METHOD, --method METHOD
                            XML RPC method name. Default is status
      -p PORT, --port PORT  Port to query. Can be integer or regular
                            expression which will be used to extract port from a
                            process name.
      -r NUM_RETRIES, --num-retries NUM_RETRIES
                            Connection retries. Default: 2

#### Configuration Examples

Call to process' XML-RPC server listening on port 8080, URL /status, RPC method get_status:

    [eventlistener:example_check]
    command=/usr/local/bin/supervisor_xmlrpc_check -g example_service -n example_check -r 3 -p 8080 -u /status -m get_status
    events=TICK_60

Call to process' XML-RPC server listening on UNIX socket:

    [eventlistener:example_check]
    command=/usr/local/bin/supervisor_xmlrpc_check -g example_service -n example_check -r 3 -s /var/run/example.sock -m get_status
    events=TICK_60

Call to process group XML-RPC servers, listening on different UNIX socket. In such
case socket directory must be specified, process socket name will be formed as <process_name>.sock:

    [eventlistener:example_check]
    command=/usr/local/bin/supervisor_xmlrpc_check -g example_service -n example_check -r 3 -S /var/run/ -m get_status
    events=TICK_60

### Memory Check

Process check based on amount of memory consumed by process.

#### CLI

    $ /usr/local/bin/supervisor_memory_check -h
    usage: supervisor_memory_check [-h] -n CHECK_NAME -g PROCESS_GROUP -m MAX_RSS
                                   [-c CUMULATIVE]

    Run memory check program.

    optional arguments:
      -h, --help            show this help message and exit
      -n CHECK_NAME, --check-name CHECK_NAME
                            Health check name.
      -g PROCESS_GROUP, --process-group PROCESS_GROUP
                            Supervisor process group name.
      -N PROCESS_NAME, --process-name PROCESS_NAME
                            Supervisor process name. Process group argument is
                            ignored if this is passed in
      -m MAX_RSS, --msx-rss MAX_RSS
                            Maximum memory allowed to use by process, KB.
      -c CUMULATIVE, --cumulative CUMULATIVE
                            Recursively calculate memory used by all process
                            children.

#### Configuration Examples

Restart process if the total amount of memory consumed by process and all its
children is greater than 100M:

    [eventlistener:example_check]
    command=/usr/local/bin/supervisor_memory_check -n example_check -m 102400 -c -g example_service
    events=TICK_60

### CPU Check

Process check based on CPU percent usage within specified time interval.

#### CLI

    $ /usr/local/bin/supervisor_cpu_check -h
    usage: supervisor_cpu_check [-h] -n CHECK_NAME -g PROCESS_GROUP -p MAX_CPU -i INTERVAL

    Run memory check program.

    optional arguments:
      -h, --help            show this help message and exit
      -n CHECK_NAME, --check-name CHECK_NAME
                            Health check name.
      -g PROCESS_GROUP, --process-group PROCESS_GROUP
                            Supervisor process group name.
      -N PROCESS_NAME, --process-name PROCESS_NAME
                            Supervisor process name. Process group argument is
                            ignored if this is passed in
      -p MAX_CPU, --max-cpu-percent MAX_CPU
                            Maximum CPU percent usage allowed to use by process
                            within time interval.
      -i INTERVAL, --interval INTERVAL
                            How long process is allowed to use CPU over threshold,
                            seconds.


#### Configuration Examples

Restart process when it consumes more than 100% CPU within 30 minutes:

    [eventlistener:example_check]
    command=/usr/local/bin/supervisor_cpu_check -n example_check -p 100 -i 1800 -g example_service
    events=TICK_60


### File Check

Process check based on file update timeout.

#### CLI

    $ /usr/local/bin/supervisor_file_check -h
    usage: supervisor_file_check [-h] -n CHECK_NAME [-g PROCESS_GROUP] [-N PROCESS_NAME] -t TIMEOUT [-x] [-f FILE]

    Run File check program.
    
    options:
      -h, --help            show this help message and exit
      -n CHECK_NAME, --check-name CHECK_NAME
                            Health check name.
      -g PROCESS_GROUP, --process-group PROCESS_GROUP
                            Supervisor process group name.
      -N PROCESS_NAME, --process-name PROCESS_NAME
                            Supervisor process name. Process group argument is ignored if this is passed in
      -t TIMEOUT, --timeout TIMEOUT
                            Timeout in seconds after no file change a process is considered dead.
      -x, --fail-on-error   Fail the health check on any error.
      -f FILEPATH, --filepath FILEPATH
                            Filepath of file to check (default:
                            %(root_directory)/%(process_group)s-%(process_name)s-%(process_pid)s-*)
      -d ROOT_DIR, --root-dir ROOT_DIR
                            Root Directory of Notification Files (default: tempfile.gettempdir())
#### Configuration Examples

Perform passive health checks on default root_directory, allowing a maximum of 30 seconds health notification delay.

    [eventlistener:example_check]
    command=/usr/local/bin/supervisor_file_check -n example_check -t 30
    events=TICK_60

#### NotificationFile Utility Class

To simplify the work with this health check module, an utility class is provided to update the notification file associated with the current supervisor subprocess and notify an heartbeat.

```python
from supervisor_checks.util import NotificationFile

tmp = NotificationFile()

while True:
    tmp.notify()
    # do some work here
```

### Complex Check

Complex check (run multiple checks at once).

#### CLI

    $ /usr/local/bin/supervisor_complex_check -h
    usage: supervisor_complex_check [-h] -n CHECK_NAME -g PROCESS_GROUP -c
                                    CHECK_CONFIG

    Run SupervisorD check program.

    optional arguments:
      -h, --help            show this help message and exit
      -n CHECK_NAME, --check-name CHECK_NAME
                            Health check name.
      -g PROCESS_GROUP, --process-group PROCESS_GROUP
                            Supervisor process group name.
      -N PROCESS_NAME, --process-name PROCESS_NAME
                            Supervisor process name. Process group argument is
                            ignored if this is passed in
      -c CHECK_CONFIG, --check-config CHECK_CONFIG
                            Check config in JSON format

#### Example configuration

Here's example configuration using memory and http checks:

    [eventlistener:example_check]
    command=/usr/local/bin/supervisor_complex_check -n example_check -g example_service -c '{"memory":{"cumulative":true,"max_rss":4194304},"http":{"timeout":15,"port":8090,"url":"\/ping","num_retries":3}}'
    events=TICK_60


## Acknowledgement

This is inspired by [Superlance](https://superlance.readthedocs.org/en/latest/) plugin package.

Though, while [Superlance](https://superlance.readthedocs.org/en/latest/) is basically the set
of feature-rich health check programs, `supervisor_checks` package is mostly focused on providing
the framework to easily implement application-specific health checks of any complexity.

## Bug reports

Please file here: <https://github.com/vovanec/supervisor_checks/issues>

Or contact me directly: <vovanec@gmail.com>

<a href="https://scan.coverity.com/projects/vovanec-supervisor_checks">
  <img alt="Coverity Scan Build Status"
       src="https://scan.coverity.com/projects/22036/badge.svg"/>
</a>
