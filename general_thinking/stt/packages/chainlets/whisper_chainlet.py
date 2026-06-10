import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import truss_chains as chains

# Path setup for local development and chain deployment
_BASETEN_WHISPER_ROOT = Path(__file__).resolve().parent.parent.parent
_WHISPER_UTILS_LIB = _BASETEN_WHISPER_ROOT / "whisper-utils"
_WHISPER_RUNTIME_LIB = _BASETEN_WHISPER_ROOT / "whisper-runtime"
# asr-chains directory (parent of chainlets/)
_ASR_CHAINS_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(_WHISPER_UTILS_LIB))
sys.path.append(str(_ASR_CHAINS_DIR))  # For whisper_chain_utils imports

from whisper_chain_utils.whisper_chain_data_types import WhisperChainletInput
from whisper_utils.constants import (
    DEFAULT_TRANSCRIBER_GPU_FOR_CHAIN,
    DEFAULT_WHISPER_MODEL,
    MAX_BATCH_SIZE,
)
from whisper_utils.data_types import WhisperResult

logger = logging.getLogger(__name__)


def load_requirements(file_path: Path) -> List[str]:
    if not file_path.exists():
        return []

    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


class Transcriber(chains.ChainletBase):
    remote_config = chains.RemoteConfig(
        docker_image=chains.DockerImage(
            base_image=chains.BasetenImage.PY310,
            apt_requirements=[
                "python3.10-venv",
                "openmpi-bin",
                "libopenmpi-dev",
                "ffmpeg",
            ],
            pip_requirements=[
                "--extra-index-url https://pypi.nvidia.com",
                "ffmpeg-python==0.2.0",
                *load_requirements(_WHISPER_RUNTIME_LIB / "requirements.txt"),
            ],
            external_package_dirs=[
                chains.make_abs_path_here(str(_WHISPER_RUNTIME_LIB)),
                chains.make_abs_path_here(str(_WHISPER_UTILS_LIB)),
                chains.make_abs_path_here(str(_ASR_CHAINS_DIR)),
            ],
        ),
        compute=chains.Compute(
            gpu=DEFAULT_TRANSCRIBER_GPU_FOR_CHAIN,
            predict_concurrency=MAX_BATCH_SIZE[DEFAULT_TRANSCRIBER_GPU_FOR_CHAIN],
        ),
        options=chains.ChainletOptions(
            env_variables={
                "HF_HOME": "/cache/org",
                "HF_HUB_CACHE": "/cache/org",
                "TLLM_LOG_LEVEL": "ERROR",
            },
            enable_debug_logs=False,
            enable_b10_tracing=False,
        ),
        assets=chains.Assets(secret_keys=["baseten_hf_access_token", "hf_access_token"]),
    )

    def __init__(
        self,
        context: chains.DeploymentContext = chains.depends_context(),
    ) -> None:
        from whisper_runtime import WhisperRT
        from whisper_utils.utilities import get_max_batch_size

        # Get the max batch size for the current GPU
        max_batch_size = get_max_batch_size()
        self.pool = ThreadPoolExecutor(max_batch_size)

        model_repo_id_override = os.environ.get("WHISPER_ENGINE_REPO_ID")
        if model_repo_id_override:
            logger.info(f"Using custom whisper engine repo: {model_repo_id_override}")

        # Prefer baseten_hf_access_token (for internal baseten-admin engine repos),
        # fall back to hf_access_token for standard customer deployments.
        # Treat blank / whitespace-only values as unset so we surface the warning
        # instead of passing a useless token into snapshot_download.
        hf_token = None
        for secret_name in ("baseten_hf_access_token", "hf_access_token"):
            try:
                candidate = context.secrets.get(secret_name)
                if isinstance(candidate, str):
                    candidate = candidate.strip()
                if candidate:
                    hf_token = candidate
                    break
            except Exception:
                pass
        if not hf_token:
            logger.warning(
                "No HF access token secret set (tried baseten_hf_access_token, hf_access_token)"
            )

        self._model = WhisperRT(
            DEFAULT_WHISPER_MODEL.value,
            max_batch_size=max_batch_size,
            hf_token=hf_token,
            model_repo_id_override=model_repo_id_override,
        )

        self._model.warmup()

    async def run_remote(self, whisper_input: WhisperChainletInput) -> WhisperResult:

        from whisper_utils.utilities import call_whisper_model

        result = await call_whisper_model(
            whisper_model=self._model,
            thread_pool_executor=self.pool,
            seg_info=whisper_input.seg_info,
            audio_chunk=whisper_input.audio_wav.array,
            fallback_chunks=whisper_input.fallback_chunks,
            whisper_params=whisper_input.whisper_params,
            asr_options=whisper_input.asr_options,
        )

        return result
