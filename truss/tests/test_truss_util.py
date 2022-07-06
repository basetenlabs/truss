import os
import time

from truss import utils


def test_max_modified():
    epoch_time = int(time.time())
    with utils.given_or_temporary_dir() as dir:
        t1 = utils.get_max_modified_time_of_dir(dir)
        assert(t1 > epoch_time)
        time.sleep(0.1)
        os.makedirs(os.path.join(dir, "test"))
        t2 = utils.get_max_modified_time_of_dir(dir)
        assert(t2 > t1)


test_max_modified()
