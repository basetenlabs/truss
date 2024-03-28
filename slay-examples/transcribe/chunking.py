import base64
import io
import shlex
import subprocess
import tempfile
from typing import Iterable, Iterator, TypeVar

import librosa
import numpy as np
import requests
from scipy.io import wavfile

TARGET_CHUNK_SIZE_SECS = 5
MODEL_SAMPLING_RATE = 16000
BLOCK_SIZE = 300
MIN_AUDIO_VALUE_FOR_SILENCE_DETECTION = 0.000001
SILENCE_DETECTION_MOVING_WINDOW_SIZE = 1000

T = TypeVar("T")


# This code is only formatted, but logically the same as in the reference.
def _last_indicating_stream(stream: Iterable[T]) -> Iterator[tuple[T, bool]]:
    prev = None
    for x in stream:
        if prev is not None:
            yield prev, False
        prev = x

    if prev is not None:
        yield prev, True


def _chunks(
    audio: np.ndarray, sample_rate: int, target_chunk_size_secs: float
) -> Iterator[tuple[int, int]]:
    target_chunk_size = int(target_chunk_size_secs * sample_rate)
    index = 0
    num_samples = len(audio)
    while index < num_samples:
        orig_index = index
        if orig_index >= num_samples - 2 * target_chunk_size:
            yield orig_index, num_samples
            return
        # Skip target chunk size
        to_split_chunk_start = orig_index + target_chunk_size
        # At least target_chunk_size samples still left, split somewhere in the next target_chunk_size
        to_split = audio[
            to_split_chunk_start : to_split_chunk_start + target_chunk_size
        ]
        # Set a lower threshold to avoid 0 values as input to log
        low_audio_values = (
            np.ones(len(to_split)) * MIN_AUDIO_VALUE_FOR_SILENCE_DETECTION
        )
        lg = np.log(np.maximum(np.abs(to_split), low_audio_values))
        moving_sum = np.convolve(
            lg, np.ones(SILENCE_DETECTION_MOVING_WINDOW_SIZE, dtype=np.float64), "same"
        )
        min_point = np.argmin(moving_sum)
        index = to_split_chunk_start + min_point
        yield orig_index, index


def _gen_chunk_stream(
    block_stream: Iterable[np.ndarray], samplerate: int, target_chunk_size: float
) -> Iterator[tuple[int, np.ndarray]]:
    index = 0
    prev_samples = np.array([])
    for block in block_stream:
        aug_block = block
        if len(prev_samples) > 0:
            aug_block = np.concatenate((prev_samples, block))
        block_chunks_indicating_last = _last_indicating_stream(
            _chunks(aug_block, samplerate, target_chunk_size)
        )
        for block_chunk, is_last in block_chunks_indicating_last:
            start, end = block_chunk
            to_transcribe = aug_block[start:end]
            if is_last:
                prev_samples = to_transcribe
                continue
            yield index, to_transcribe
            index += len(to_transcribe)

    # Process left over, last prev_samples
    for start, end in _chunks(prev_samples, samplerate, target_chunk_size):
        to_transcribe = prev_samples[start:end]
        yield index, to_transcribe
        index += len(to_transcribe)


# Code differeing from reference app ###################################################


def _download_and_extract_audio(media_url: str, wave_file_path: str) -> None:
    with (
        tempfile.NamedTemporaryFile(prefix="video_") as download_file,
        requests.get(media_url, stream=True) as download_stream,
    ):
        if download_stream.status_code == 200:
            for i, chunk in enumerate(
                download_stream.iter_content(chunk_size=32 * 1024)
            ):
                download_file.write(chunk)
        else:
            raise IOError(f"Failed to download the video `{media_url}`")

        download_file.flush()

        ffmpeg_process = subprocess.run(
            shlex.split(
                f"ffmpeg -y -i {download_file.name} -f wav "
                f"-ar {MODEL_SAMPLING_RATE} {wave_file_path}"
            )
        )
        ffmpeg_process.check_returncode()


def download_and_generate_chunks(media_url: str) -> Iterator[tuple[int, str]]:
    with tempfile.NamedTemporaryFile(prefix="wav_") as wav_file:
        _download_and_extract_audio(media_url, wav_file.name)
        # samplerate_float = librosa.get_samplerate(wav_file_path)
        # if samplerate_float % 1:
        #     raise ValueError(f"Non-int sample rate: `{samplerate_float}`")
        # samplerate = int(samplerate_float)
        # assert samplerate == MODEL_SAMPLING_RATE
        samplerate = MODEL_SAMPLING_RATE
        block_stream = librosa.stream(
            wav_file.name,
            block_length=BLOCK_SIZE,
            frame_length=samplerate,
            hop_length=samplerate,
        )
        for index, audio_chunk in _gen_chunk_stream(
            block_stream, samplerate, TARGET_CHUNK_SIZE_SECS
        ):
            # for index, chunk in self._chunk_audio("/tmp/test.wav"):
            audio_b64 = _numpy_to_b64wav(audio_chunk, MODEL_SAMPLING_RATE)
            yield index, audio_b64


def _numpy_to_b64wav(audio_data: np.ndarray, sample_rate: int) -> str:
    bytes_io = io.BytesIO()
    wavfile.write(bytes_io, sample_rate, audio_data)
    bytes_io.seek(0)  # Go to the beginning of the BytesIO object
    encoded_audio = base64.b64encode(bytes_io.read()).decode("utf-8")
    return encoded_audio
