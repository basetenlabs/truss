import logging
import os
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .assets import MEL_FILTERS_PATH
from .configs import *

BASE_PATH = os.path.dirname(__file__)

logger = logging.getLogger(__name__)


def pad_or_trim(
    array: Union[torch.Tensor, np.array], length: int = N_SAMPLES, *, axis: int = -1
) -> torch.Tensor:
    """
    Pad or trim the audio array to the specified length.

    Args:
        array (torch.Tensor | np.array): The input audio array.
        length (int, optional): The target length of the audio array. Defaults to N_SAMPLES.
        axis (int, optional): The axis along which to pad or trim. Defaults to -1.

    Returns:
        torch.Tensor: The padded or trimmed audio array.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            logger.warning("Audio is longer than expected. Trimming...")
            array = array.index_select(dim=axis, index=torch.arange(length, device=array.device))

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            logger.warning("Audio is longer than expected. Trimming...")
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


class TorchSTFT(nn.Module):
    """
    A class that encapsulates the Short-Time Fourier Transform (STFT) using PyTorch.

    Args:
        n_fft (int): The size of FFT.
        hop_length (int): The hop (or stride) length.
    """

    def __init__(self, n_fft, hop_length):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length

        window = torch.hann_window(n_fft)
        self.register_buffer("window", window)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the STFT on the input tensor.

        Args:
            x (torch.Tensor): The input waveform tensor.

        Returns:
            torch.Tensor: The complex STFT result.
        """
        return torch.stft(x, self.n_fft, self.hop_length, window=self.window, return_complex=True)


class LogMelSpectrogram(nn.Module):
    """
    A class that computes the log-mel spectrogram from an audio waveform.

    Args:
        n_mels (int): The number of mel filter banks.
        n_fft (int): The size of FFT.
        hop_length (int): The hop (or stride) length.
        padding (int): The amount of padding to apply to the input waveform.
    """

    def __init__(self, n_mels, n_fft=N_FFT, hop_length=HOP_LENGTH):
        super().__init__()

        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = hop_length

        mel_filters = np.load(MEL_FILTERS_PATH)
        mel_filters = torch.from_numpy(mel_filters[f"mel_{n_mels}"])
        self.register_buffer("mel_filters", mel_filters)

        self.stft = TorchSTFT(n_fft, hop_length)

    def get_seq_len(self, seq_len: torch.Tensor) -> torch.Tensor:
        """
        Calculate the sequence length after applying the STFT.

        Args:
            seq_len (torch.Tensor): The original sequence length.

        Returns:
            torch.Tensor: The sequence length after STFT.
        """
        seq_len = torch.floor(seq_len / self.hop_length)
        return seq_len.to(dtype=torch.long)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        seq_len: torch.Tensor,
        begin_padding_seconds: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform the log-mel spectrogram transformation on the input waveform.

        Args:
            x (torch.Tensor): The input waveform tensor.
            seq_len (torch.Tensor): The original sequence length tensor.

        Returns:
            torch.Tensor: The log-mel spectrogram.
            torch.Tensor: The sequence length after STFT.
        """
        seq_len = self.get_seq_len(seq_len.float())

        if begin_padding_seconds > 0:
            padding_samples = int(SAMPLE_RATE * begin_padding_seconds)
            x = F.pad(x[:, :-padding_samples], (padding_samples, 0), mode="replicate")

        x = self.stft(x)

        x = x[..., :-1].abs() ** 2
        x = self.mel_filters @ x  # mels

        x = torch.clamp(x, min=1e-10).log10()  # log_mels
        x = torch.maximum(x, torch.amax(x, dim=(1, 2), keepdims=True) - 8.0)  # clip
        x = (x + 4.0) / 4.0  # scale

        return x, seq_len
