import os
import uuid
from contextlib import contextmanager


def sync_leader_work(work_fn):
    """
    sync_leader_work provides a useful utility for synchronizing work
    for the leader.
    """
    sync_id = str(uuid.uuid4())
    if os.environ.get("BT_NODE_RANK") == "0":
        begin_write(sync_id)  # writes to shared mounted storage
        work_fn()
        end_write(sync_id)
    else:
        wait_for_begin_write(sync_id)  # reads from shared_mounted storage
        wait_for_end_write(sync_id)


@contextmanager
def baseten_cache_read_context():
    """
    baseten_cache_read_context guarantees successful reads from the cache
    when interacting with huggingface. It sets the HF_LOCAL_FILES_ONLY env var
    to true to avoid metadata writes that HF might perform during reads.
    """

    class _BasetenCacheReadContext:
        def __init__(self):
            self.current_val = os.environ.get("HF_LOCAL_FILES_ONLY")

        def __enter__(self):
            os.environ["HF_LOCAL_FILES_ONLY"] = "true"

        def __exit__(self, exc_type, exc_val, exc_tb):
            os.environ["HF_LOCAL_FILES_ONLY"] = self.current_val
            return False

    return _BasetenCacheReadContext()


with baseten_cache_read_context() as ctx:
    model = modelxyz.from_pretrained("bert-base-uncased")
