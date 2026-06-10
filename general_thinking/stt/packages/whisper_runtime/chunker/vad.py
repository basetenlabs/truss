from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..assets import ASSETS_DIR

BASE_PATH = Path(__file__).parent.parent


class VAD:
    """
    Voice Activity Detection (VAD) class that uses Marblenet VAD to detect speech in audio signals.

    Attributes:
        device (str): The device to run the model on ('cpu' or 'cuda').
        chunk_size (float): The size of each audio chunk in seconds.
        margin_size (float): The margin size in seconds.
        frame_size (float): The size of each frame in seconds.
        batch_size (int): The batch size for processing.
        sampling_rate (int): The sampling rate of the audio signal.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        chunk_size: float = 15.0,
        margin_size: float = 1.0,
        frame_size: float = 0.02,
        batch_size: int = 4,
        sampling_rate: int = 16000,
    ):
        """
        Initializes the VAD class with the given parameters.

        Args:
            device (str, optional): The device to run the model on. Defaults to None.
            chunk_size (float, optional): The size of each audio chunk in seconds. Defaults to 15.0.
            margin_size (float, optional): The margin size in seconds. Defaults to 1.0.
            frame_size (float, optional): The size of each frame in seconds. Defaults to 0.02.
            batch_size (int, optional): The batch size for processing. Defaults to 4.
            sampling_rate (int, optional): The sampling rate of the audio signal. Defaults to 16000.
        """
        self.sampling_rate = sampling_rate

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        if self.device == "cpu":
            # This is a JIT Scripted model of Nvidia's NeMo Framewise Marblenet Model: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/vad_multilingual_frame_marblenet
            vad_pp_path = ASSETS_DIR / "vad_pp_cpu.ts"
            model_path = ASSETS_DIR / "frame_vad_model_cpu.ts"
        else:
            vad_pp_path = ASSETS_DIR / "vad_pp_gpu.ts"
            model_path = ASSETS_DIR / "frame_vad_model_gpu.ts"

        self.vad_pp = torch.jit.load(vad_pp_path).to(self.device)
        self.vad_pp.eval()
        self.vad_model = torch.jit.load(model_path).to(self.device)
        self.vad_model.eval()

        self.batch_size = batch_size
        self.frame_size = frame_size
        self.chunk_size = chunk_size
        self.margin_size = margin_size

        self._init_params()

    def _init_params(self) -> None:
        """
        Initializes internal parameters based on the given attributes.
        """
        self.signal_chunk_len = int(self.chunk_size * self.sampling_rate)
        self.signal_stride = int(
            self.signal_chunk_len - 2 * int(self.margin_size * self.sampling_rate)
        )

        self.margin_logit_len = int(self.margin_size / self.frame_size)
        self.signal_to_logit_len = int(self.frame_size * self.sampling_rate)

        self.vad_pp.to(self.device)
        self.vad_model.to(self.device)

    def update_params(self, params: Dict[str, float]) -> None:
        """
        Updates the parameters of the VAD class and re-initializes internal parameters.

        Args:
            params (dict): A dictionary of parameters to update.
        """
        for key, value in params.items():
            setattr(self, key, value)

        self._init_params()

    def prepare_input_batch(self, audio_signal: np.ndarray) -> Tuple[List[np.ndarray], List[int]]:
        """
        Prepares the input batch for the VAD model.

        Args:
            audio_signal (numpy.ndarray): The audio signal to process.

        Returns:
            tuple: A tuple containing the input signal and input signal length.
        """
        input_signal = []
        input_signal_length = []
        for s_idx in range(0, len(audio_signal), self.signal_stride):
            _signal = audio_signal[s_idx : s_idx + self.signal_chunk_len]
            _signal_len = len(_signal)
            input_signal.append(_signal)
            input_signal_length.append(_signal_len)

            if _signal_len < self.signal_chunk_len:
                input_signal[-1] = np.pad(
                    input_signal[-1], (0, self.signal_chunk_len - _signal_len)
                )
                break

        return input_signal, input_signal_length

    @torch.amp.autocast("cuda")
    @torch.no_grad()
    def forward(self, input_signal: List[np.ndarray], input_signal_length: List[int]) -> np.ndarray:
        """
        Performs forward pass on the input signal using the VAD model.

        Args:
            input_signal (list): The input signal to process.
            input_signal_length (list): The lengths of the input signals.

        Returns:
            numpy.ndarray: The speech probabilities for each frame.
        """
        all_logits = []
        for s_idx in range(0, len(input_signal), self.batch_size):
            input_signal_pt = torch.stack(
                [
                    torch.tensor(_, device=self.device)
                    for _ in input_signal[s_idx : s_idx + self.batch_size]
                ]
            )
            input_signal_length_pt = torch.tensor(
                input_signal_length[s_idx : s_idx + self.batch_size], device=self.device
            )

            x, x_len = self.vad_pp(input_signal_pt, input_signal_length_pt)
            logits = self.vad_model(x, x_len)

            for _logits, _len in zip(logits, input_signal_length_pt):
                all_logits.append(_logits[: int(_len / self.signal_to_logit_len)])

        if len(all_logits) > 1 and self.margin_logit_len > 0:
            all_logits[0] = all_logits[0][: -self.margin_logit_len]
            all_logits[-1] = all_logits[-1][self.margin_logit_len :]

            for i in range(1, len(all_logits) - 1):
                all_logits[i] = all_logits[i][self.margin_logit_len : -self.margin_logit_len]

        all_logits = torch.concatenate(all_logits)
        all_logits = torch.softmax(all_logits, dim=-1)

        return all_logits[:, 1].detach().cpu().numpy()

    def __call__(self, audio_signal: np.ndarray) -> np.ndarray:
        """
        Processes the audio signal and returns the voice activity detection times.

        Args:
            audio_signal (numpy.ndarray): The audio signal to process.

        Returns:
            numpy.ndarray: An array of voice activity detection times and probabilities.
        """
        audio_duration = len(audio_signal) / self.sampling_rate

        input_signal, input_signal_length = self.prepare_input_batch(audio_signal)
        speech_probs = self.forward(input_signal, input_signal_length)

        vad_times = []
        for idx, prob in enumerate(speech_probs):
            s_time = idx * self.frame_size
            e_time = min(audio_duration, (idx + 1) * self.frame_size)

            if s_time >= e_time:
                break

            vad_times.append([prob, s_time, e_time])

        return np.array(vad_times)
