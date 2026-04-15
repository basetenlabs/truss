"""Stub out heavy ML and deploy dependencies so test modules can be imported
without the full worker (ms-swift, vllm, megatron, CUDA torch) or deploy
stack (truss internals that require watchfiles, boto3, etc.).
"""

import sys
from unittest.mock import MagicMock

_HEAVY_MODULES = [
    # ML worker dependencies
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
    "swift.tuners",
    "vllm",
    # Deploy-stack dependencies pulled in by trainers.client (only needed for
    # deploy=True; safe to stub for all SDK unit/integration tests)
    "truss_train",
    "truss_train.definitions",
    "truss_train.public_api",
    "truss.base",
    "truss.base.truss_config",
]

for _mod in _HEAVY_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
