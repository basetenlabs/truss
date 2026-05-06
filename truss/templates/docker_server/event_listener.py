#!/usr/bin/env python3
"""Supervisord event listener that filters /metrics scrape logs from process output.

Analogous to _MetricsFilter in shared/log_config.py which filters /metrics from
model server logs. For docker_server deployments we don't control the custom
server's logging, so we filter at the supervisord output level instead.
"""

import sys


def main():
    with open("/proc/1/fd/1", "w") as container_stdout:
        while True:
            sys.stdout.write("READY\n")
            sys.stdout.flush()

            header_line = sys.stdin.readline()
            if not header_line:
                break

            headers = dict(x.split(":") for x in header_line.split())
            data = sys.stdin.read(int(headers["len"]))

            # PROCESS_LOG_STDOUT payload: event-specific header line + log data
            _, _, log_data = data.partition("\n")

            for line in log_data.splitlines(True):
                if "/metrics" not in line:
                    container_stdout.write(line)
                    container_stdout.flush()

            sys.stdout.write("RESULT 2\nOK")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
