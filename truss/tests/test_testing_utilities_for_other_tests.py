# This file contains shared code to be used in other tests
# TODO(pankaj): Using a tests file for shared code is not ideal, we should
# move it to a regular file. This is a short term hack.

import time
from contextlib import contextmanager

from truss.build import kill_all
from truss.constants import TRUSS
from truss.docker import get_containers


@contextmanager
def ensure_kill_all():
    try:
        yield
    finally:
        kill_all_with_retries()


def kill_all_with_retries(num_retries: int = 10):
    kill_all()
    attempts = 0
    while attempts < num_retries:
        containers = get_containers({
            TRUSS: True
        })
        if len(containers) == 0:
            return
        attempts += 1
        time.sleep(1)
