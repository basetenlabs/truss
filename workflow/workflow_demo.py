"""
Goals:
* Python code should be real truthful python i.e. if two symbols aren't the same thing
  (e.g. on a local def and another a deployed objects) they should not appear to be the
  same.
* Code completion on args/types of deployed objects.
* Static type checking of deployed objects.
* Way to setup persistent resources for deployed objects.
* Imperative DX for worklow.
    - Orchestrator might become bottleneck e.g. if huge fan-in of intermediate results.
    - Suboptimal communication pattern.
    - Unclear mix of execution options/params and input data.
* Testability?
* Allow processor to call directly another processor?
"""
import abc
from typing import Any, TypeAlias

from pydantic import BaseModel
from truss.workflow import api


# If the symbols `chunking` and `stiching` are not actually the same objects inside
# workflow and defined above (becaus there was a deployment step in between) they
# should not look as if they are the same here. I.e. we need
@workflow(
    name="End to End Creator Pipeline",  # Optional to just use inline; If provided, can be used outside workflow as well
)
def run_transcription_on_really_long_things(wkt_in: workflowInput):
    urls = chunking(wkt_in.url, batch_size=8, parallel=True)
    # what is the parallilization mechanism
    transcripts = hosted.model_by_name("Faster-Whisper-v3", batch=True, parallel=True)(
        urls
    )  # or by id or whatever
    # hosted.workflow_by_name()
    if len(transcripts) > 1:
        # return stitching()(transcripts)
        return stitch_handle.process(transcripts)
    return transcripts[0]
