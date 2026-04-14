"""Stub out heavy ML dependencies so server modules can be imported without the
full worker installation (ms-swift, vllm, megatron, CUDA torch).
"""

import sys
from unittest.mock import MagicMock

_HEAVY_MODULES = [
    "swift",
    "swift.arguments",
    "swift.infer_engine",
    "swift.infer_engine.protocol",
    "swift.model.register",
    "swift.pipelines.infer.rollout",
    "swift.rlhf_trainers",
    "swift.rlhf_trainers.utils",
    "swift.rlhf_trainers.vllm_client",
    "swift.template",
    "swift.template.register",
    "vllm",
]

for _mod in _HEAVY_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
