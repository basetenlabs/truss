import logging
import sys
from pathlib import Path

import truss_chains as chains
from truss import truss_config

# Path setup for local development and chain deployment
_BASETEN_WHISPER_ROOT = Path(__file__).resolve().parent.parent.parent
_WHISPER_UTILS_LIB = _BASETEN_WHISPER_ROOT / "whisper-utils"
# asr-chains directory (parent of chainlets/)
_ASR_CHAINS_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(_WHISPER_UTILS_LIB))
sys.path.append(str(_ASR_CHAINS_DIR))  # For whisper_chain_utils and diart imports

from whisper_chain_utils.whisper_chain_data_types import DiarizationInput, DiarizationOutput

logger = logging.getLogger(__name__)


class Diarizer(chains.ChainletBase):
    remote_config = chains.RemoteConfig(
        docker_image=chains.DockerImage(
            base_image=chains.BasetenImage.PY310,
            pip_requirements=[
                "pyannote-audio==4.0.1",
                "async-batcher==0.2.2",
                "omegaconf>=2.0.0",
            ],
            apt_requirements=[
                "ffmpeg",
            ],
            external_package_dirs=[
                chains.make_abs_path_here(str(_WHISPER_UTILS_LIB)),
                chains.make_abs_path_here(str(_ASR_CHAINS_DIR)),
            ],
        ),
        compute=chains.Compute(
            gpu=truss_config.Accelerator.L4,
            predict_concurrency=50,
        ),
        assets=chains.Assets(secret_keys=["hf_access_token"]),
        options=chains.ChainletOptions(env_variables={"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"}),
    )

    _context: chains.DeploymentContext

    def __init__(self, context: chains.DeploymentContext = chains.depends_context()):
        import torch
        from diart.diarization import SpeakerDiarizationConfig
        from diart.embedding import OverlapAwareSpeakerEmbedding
        from diart.models import EmbeddingModel, SegmentationModel
        from diart.segmentation import SpeakerSegmentation

        # for image streaming optimization
        from pyannote.core import SlidingWindow, SlidingWindowFeature
        from whisper_chain_utils.diarizer_async_batcher import DiarizerBatchProcessor

        ### Params
        self.duration = 5.0
        self.step = 0.5
        self.sample_rate = 16000
        self.max_async_batch_size = 20

        self.diarization_model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hf_token = context.secrets["hf_access_token"]
        if not hf_token:
            raise ValueError("HF access token is required")

        self.segmentation_model = SegmentationModel.from_pyannote(
            "pyannote/segmentation", use_hf_token=hf_token
        ).to(self.diarization_model_device)

        self.embedding_model = EmbeddingModel.from_pyannote(
            "pyannote/embedding", use_hf_token=hf_token
        ).to(self.diarization_model_device)

        # torch compile models

        self._config = SpeakerDiarizationConfig(
            segmentation=self.segmentation_model,
            embedding=self.embedding_model,
            device=self.diarization_model_device,
            latency="max",
            step=self.step,
            duration=self.duration,
        )

        self.segmentation = SpeakerSegmentation(
            self._config.segmentation, self.diarization_model_device
        )

        self.embedding = OverlapAwareSpeakerEmbedding(
            self._config.embedding,
            self._config.gamma,
            self._config.beta,
            norm=1,
            normalize_weights=self._config.normalize_embedding_weights,
            device=self._config.device,
        )

        ## warmup
        self.segmentation.model(torch.randn(10, 1, 80000).to(self.diarization_model_device))
        self.embedding.embedding.model(
            torch.randn(30, 1, 80000).to(self.diarization_model_device),
            torch.randn(30, 293).to(self.diarization_model_device),
        )

        self.diarizer_batch_processor = DiarizerBatchProcessor(
            segmentation_model=self.segmentation,
            embedding_model=self.embedding,
            max_batch_size=self.max_async_batch_size,
            max_queue_time=1,
        )

    async def run_remote(self, diarization_input: DiarizationInput) -> DiarizationOutput:
        import numpy as np
        import torch

        audio_array = diarization_input.audio_wav.array
        required_samples = int(self.duration * self.sample_rate)
        step_samples = int(self.step * self.sample_rate)
        audio_length = audio_array.shape[0]

        # Vectorized preprocessing: calculate chunk indices efficiently
        if audio_length < required_samples:
            # Handle edge case: audio shorter than required duration
            # Pad to required_samples and create single chunk
            batch_np = np.pad(audio_array, (0, required_samples - audio_length), mode="constant")
            batch_np = batch_np.reshape(1, -1)
        else:
            # Calculate number of chunks that fit
            max_start = audio_length - required_samples
            num_chunks = max(1, (max_start // step_samples) + 1)

            # Create chunk start indices using numpy (vectorized)
            chunk_starts = np.arange(0, max_start + 1, step_samples)[:num_chunks]

            # Vectorized chunk extraction using advanced indexing
            # Create indices for all chunks at once: (num_chunks, required_samples)
            chunk_indices = chunk_starts[:, None] + np.arange(required_samples)
            batch_np = audio_array[chunk_indices]

            # Ensure last chunk has correct size (pad if needed)
            if batch_np.shape[1] < required_samples:
                padding = required_samples - batch_np.shape[1]
                batch_np = np.pad(batch_np, ((0, 0), (0, padding)), mode="constant")

        # Convert to torch tensor and move to GPU in one operation
        # Reshape to (batch, samples, channels) format expected by models
        batch = torch.from_numpy(batch_np).float().unsqueeze(-1)  # Add channel dimension
        batch = batch.to(self.diarization_model_device)

        expected_num_samples = int(self.duration * self.sample_rate)
        assert (
            batch.shape[1] == expected_num_samples
        ), f"Expected {expected_num_samples} samples per chunk, but got {batch.shape[1]}"

        # Process batch (tensors stay on GPU)
        results = await self.diarizer_batch_processor.process(batch)

        segmentations = results[0]
        embeddings = results[1]

        # Convert to numpy only at the end, when creating output
        # Use .cpu() first to move to CPU, then .numpy() to avoid blocking
        diarization_output = DiarizationOutput(
            segmentation=segmentations.cpu().numpy(),
            embedding=embeddings.cpu().numpy(),
        )

        return diarization_output
