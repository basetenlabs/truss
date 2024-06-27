import asyncio
import base64
import datetime
import io
import itertools
import logging
import signal
import wave
from asyncio import subprocess
from typing import AsyncIterator, Iterable, Optional, TypeVar

import data_types
import httpx
import numpy as np
from truss_chains import utils

_WAV_HEADER_NUM_BYTES = 78
# These flags are for debugging / dev only.
_INJECT_ERRORS = False
_DEBUG_PLOTS = False


def _format_time_ffmpeg(seconds: float) -> str:
    duration = datetime.timedelta(seconds=seconds)
    days, remainder = divmod(duration.total_seconds(), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds % 1) * 1000)
    formatted_time = (
        f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}.{milliseconds:03}"
    )
    return formatted_time


def generate_time_chunks(
    duration_sec: float, macro_chunk_size_sec: int, overlap_sec: int
) -> list[data_types.ChunkInfo]:
    step_sec = macro_chunk_size_sec + overlap_sec
    start_times = np.arange(0, duration_sec, macro_chunk_size_sec)
    end_times = np.minimum(start_times + step_sec, duration_sec)
    chunks = []
    for i, (start, end) in enumerate(zip(start_times, end_times)):
        chunks.append(
            data_types.ChunkInfo(
                start_time_sec=start,
                end_time_sec=end,
                start_time_str=_format_time_ffmpeg(float(start)),
                duration_sec=end - start,
                is_last=False,
                macro_chunk=i,
            )
        )
    chunks[-1].is_last = True
    return chunks


async def assert_media_supports_range_downloads(
    httpx_client: httpx.AsyncClient, media_url: str
) -> None:
    # TODO: specific logging if 404 errors occur.
    ok = False
    try:
        head_response = await httpx_client.head(media_url)
        if "bytes" in head_response.headers.get("Accept-Ranges", ""):
            ok = True
        # Check by making a test range request to see if '206' is returned.
        range_header = {"Range": "bytes=0-0"}
        range_response = await httpx_client.get(media_url, headers=range_header)
        ok = range_response.status_code == 206
    except httpx.HTTPError as e:
        logging.error(f"Error checking URL: {e}")

    if not ok:
        raise NotImplementedError(f"Range downloads unsupported for `{media_url}`.")


async def query_source_length_secs(media_url: str) -> float:
    cmd = (
        "ffprobe -v error -show_entries format=duration -of "
        f'default=noprint_wrappers=1:nokey=1 "{media_url}"'
    )
    stderr = None
    try:
        proc = await subprocess.create_subprocess_shell(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        duration = float(stdout.strip())
        return duration
    except Exception as e:
        if stderr:
            error_str = f".\nFFMPEG stderr:\n{stderr.decode()}"
        else:
            error_str = ""
        raise ValueError(
            f"Could not retrieve source duration for `{media_url}`, error: {error_str}."
        ) from e


def _dbg_show_waveform(data, x_marker):
    import matplotlib

    matplotlib.use("qtagg")
    from matplotlib import pyplot as plt

    plt.ioff()
    plt.figure()
    plt.plot(data)
    plt.vlines(x_marker, 0, 20000, colors=["red"])
    plt.grid()


def _wav_block_to_b64(wav_block: bytes, wav_info: data_types.WavInfo) -> str:
    """Encodes raw wave form byte data (without metadat) to b64 encoded wave file."""
    with io.BytesIO() as out_buffer:
        with wave.open(out_buffer, "wb") as out_wav:
            out_wav.setnchannels(wav_info.num_channels)
            out_wav.setsampwidth(wav_info.bytes_per_sample)
            out_wav.setframerate(wav_info.sampling_rate_hz)
            out_wav.writeframes(wav_block)
        out_buffer.seek(0)  # Go to the beginning of the BytesIO object
        if _INJECT_ERRORS:
            utils.random_fail(0.03, "wave_b64_encoding")
        return base64.b64encode(out_buffer.read()).decode("utf-8")


async def _extract_wav_info(
    stream: asyncio.StreamReader,
) -> tuple[data_types.WavInfo, bytes]:
    try:
        # Wave headers data was typically 78 bytes, add some extra margin.
        initial_data = await stream.readexactly(_WAV_HEADER_NUM_BYTES + 128)
    except asyncio.IncompleteReadError as e:
        # If not enough data was read to parse at least the header, another
        # exception will be raised below when trying to do that.
        initial_data = e.partial
        if len(initial_data) < _WAV_HEADER_NUM_BYTES:
            raise ValueError(
                f"Could not read at least `{_WAV_HEADER_NUM_BYTES}` "
                "from the audio byte stream. This indicates the source file "
                "is broken or FFMPEG cannot download and convert it. Check for other "
                "error messages."
            ) from e

    with io.BytesIO(initial_data) as initial_io:
        with wave.open(initial_io, "rb") as wf:
            num_channels = wf.getnchannels()
            sampling_rate_hz = wf.getframerate()
            bytes_per_sample = wf.getsampwidth()
            # Get the position where audio data begins by reading a frame.
            wf.readframes(1)
            position_after_header = initial_io.tell() - (
                bytes_per_sample * num_channels
            )
    # Separate the initial audio data from the header.
    initial_audio_data = initial_data[position_after_header:]
    wav_info = data_types.WavInfo(
        num_channels=num_channels,
        sampling_rate_hz=sampling_rate_hz,
        bytes_per_sample=bytes_per_sample,
    )
    logging.info(
        f"wav_info={wav_info} position_after_header={position_after_header} "
        f"len(initial_data)={len(initial_data)}"
    )
    if _INJECT_ERRORS:
        utils.random_fail(0.02, "wave_info")
    return wav_info, initial_audio_data


def _find_silent_split_point(
    wav_block: bytes,
    smoothing_num_samples: int,
    wav_info: data_types.WavInfo,
) -> int:
    # Note: this function uses both indices w.r.t to the bytes array and the numpy
    # array, with `bytes_per_sample` as a conversion factor. To avoid confusion
    # the former are suffixed with `bytes` and numpy-based indices are with `_np`.
    if len(wav_block) < 3:
        # This is just a safeguard, we should skip silence detection fo chunks smaller
        # than 1/2 of maximal micro chunk length on the call site.
        return len(wav_block)

    half_index_bytes = len(wav_block) // 2
    # Make sure we always split full samples.
    half_index_bytes += half_index_bytes % wav_info.bytes_per_sample
    assert wav_info.bytes_per_sample == 2  # For int16.
    data = np.frombuffer(wav_block[half_index_bytes:], dtype=np.int16)
    assert data.ndim == 1
    # Offset by one for correct index calculation.
    cumsum = np.zeros(shape=len(data) + 1, dtype=np.float32)
    # Cumsum is a faster way to calculate convolution with box filter.
    np.cumsum(abs(data), out=cumsum[1:])
    smoothed_data = cumsum[smoothing_num_samples:] - cumsum[:-smoothing_num_samples]
    min_index_np = np.argmin(smoothed_data)
    split_index_np = min_index_np + smoothing_num_samples // 2
    split_index_bytes = split_index_np * wav_info.bytes_per_sample + half_index_bytes

    if _DEBUG_PLOTS:
        dbg_data = np.frombuffer(wav_block, dtype=np.int16)
        _dbg_show_waveform(
            abs(dbg_data),
            split_index_np + half_index_bytes / wav_info.bytes_per_sample,
        )
        # _dbg_show_waveform(data, split_index)
        # _dbg_show_waveform(smoothed_data, split_index)
        from matplotlib import pyplot as plt

        plt.show(block=True)

    return int(split_index_bytes)


class DownloadSubprocess:
    """Contextmanager reading subprocess streams asynchronously and
    properly handling errors.

    Given a media URL and time boundaries, emits on the `wav_stream`-attribute
    an async byte stream of the extracted and resampled mono-channel waveform.
    """

    media_url: str
    chunk_info: data_types.ChunkInfo
    _ffmpeg_command: str
    _process: Optional[subprocess.Process]

    def __init__(
        self,
        media_url: str,
        chunk_info: data_types.ChunkInfo,
        wav_sampling_rate_hz: int,
    ) -> None:
        # -af "pan=mono|c0=0.5*c0+0.5*c1": Applies an audio filter to downmix to mono
        #  by averaging the channels. The pan filter is used here to control the mix.
        #  c0=0.5*c0+0.5*c1 takes the first channel (c0) and averages it with the second
        #  channel (c1). For sources with more than two channels, adjust the formula to
        #  include them accordingly.
        ffmpeg_command = (
            f"ffmpeg -i {media_url} "
            f"-ss {chunk_info.start_time_str} "  # seek time stamp.
            f"-t {chunk_info.duration_sec} "
            f"-vn "  # Disables video recording.
            f"-acodec pcm_s16le "  # Audio codec: PCM signed 16-bit little endian.
            f"-ar {wav_sampling_rate_hz} "
            f"-ac 1 "  # -ar: Sets the audio sample rate.
            f"-af 'pan=mono|c0=0.5*c0+0.5*c1' "  # Average channels to mono.
            f"-f wav "  # -f wav: Specifies the output format to be WAV.
            f"-"  # -: write output to stdout.
        )
        self.media_url = media_url
        self.chunk_info = chunk_info
        self._ffmpeg_command = ffmpeg_command
        self._process = None

    async def __aenter__(self) -> "DownloadSubprocess":
        self._process = await subprocess.create_subprocess_shell(
            self._ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert self._process.stdout is not None
        return self

    async def __aexit__(self, exc_type, exc, tb):
        assert self._process is not None
        logging.debug(f"Exiting context, exc_type={exc_type}, exc={exc}")
        if self._process.returncode is None:
            logging.debug("Sending sigterm to sub-process.")
            self._process.terminate()

        try:
            logging.debug("Waiting for sub-process.")
            await asyncio.wait_for(self._process.wait(), 5.0)
        except asyncio.TimeoutError as e:
            logging.debug("Waiting timed out, killing sub-process.")
            self._process.kill()
            await self._process.wait()  # Wait again after killing
            stderr = (
                (await self._process.stderr.read()).decode()
                if self._process.stderr
                else "No stderr available."
            )
            raise ChildProcessError(
                "FFMPEG hangs after terminating. Stderr:\n" f"{stderr}"
            ) from e

        logging.debug(f"return code={self._process.returncode}.")
        # The sub-proces did not crash (it succeeded, or we terminated it) - but
        # there was an exception from the within the context block.
        if exc and self._process.returncode in (0, -signal.SIGTERM, -signal.SIGKILL):
            logging.debug("Re-raising exception from main process.")
            raise exc

        # The subprocess crashed.
        if self._process.returncode != 0:
            logging.debug("Raising exception from failed sub-process.")
            stderr = (
                (await self._process.stderr.read()).decode()
                if self._process.stderr
                else "No stderr available."
            )
            raise ChildProcessError(
                "FFMPEG error during source download and "
                f"wav extraction. Stderr:\n{stderr}"
            ) from exc  # In case there was also an error in the context-block, chain.

        # E.g. there was an exception AND we couldn't terminate the sub-process.
        elif exc:
            logging.debug(f"Handling of exception not otherwise covered: {exc}")
            raise exc

    @property
    def wav_stream(self) -> asyncio.StreamReader:
        assert self._process is not None
        stdout = self._process.stdout
        assert stdout is not None
        return stdout


async def wav_chunker(
    params: data_types.TranscribeParams,
    download: DownloadSubprocess,
) -> AsyncIterator[tuple[data_types.ChunkInfo, str]]:
    """Consumes the download stream and yields small chunks of b64-encoded wav."""
    wav_info, initial_audio_data = await _extract_wav_info(download.wav_stream)
    if params.macro_chunk_overlap_sec > 0:
        # TODO: find consistent split point chunks in overlap area, skip last.
        raise NotImplementedError("macro_chunk_overlap_sec")
    assert wav_info.sampling_rate_hz == params.wav_sampling_rate_hz
    assert wav_info.num_channels == 1
    micro_chunk_num_bytes = wav_info.get_chunk_size_bytes(params.micro_chunk_size_sec)
    wav_block_buffer = initial_audio_data
    # We don't know the exact number of chunks depending on how silence detection
    # splits the file, but we can approximate a number assuming it is equally
    # likely to split anywhere in the second half of the maximal micro chunk length.
    # I.e. each chunk is on average 3/4 of the maximal size.
    approx_num_micro_chunks = int(
        np.ceil(download.chunk_info.duration_sec / params.micro_chunk_size_sec * 4 / 3)
    )
    start_time_sec = download.chunk_info.start_time_sec
    for i in itertools.count():
        required_read = micro_chunk_num_bytes - len(wav_block_buffer)
        try:
            new_data = await download.wav_stream.readexactly(required_read)
        except asyncio.IncompleteReadError as e:
            new_data = e.partial
            assert len(new_data) % wav_info.bytes_per_sample == 0

        wav_block_buffer += new_data
        if not wav_block_buffer:  # Stream is fully consumed.
            break

        if len(wav_block_buffer) < micro_chunk_num_bytes / 2:
            # Don't split short (last) block.
            split_index = len(wav_block_buffer)
        else:
            split_index = _find_silent_split_point(
                wav_block_buffer,
                params.silence_detection_smoothing_num_samples,
                wav_info,
            )

        micro_chunk = wav_block_buffer[:split_index]
        duration_sec = wav_info.get_chunk_duration_sec(len(micro_chunk))
        end_time_sec = start_time_sec + duration_sec
        seg_info = data_types.ChunkInfo(
            start_time_sec=start_time_sec,
            end_time_sec=end_time_sec,
            duration_sec=end_time_sec - start_time_sec,
            start_time_str=_format_time_ffmpeg(start_time_sec),
            is_last=download.chunk_info.is_last
            and end_time_sec == download.chunk_info.end_time_sec,
            macro_chunk=download.chunk_info.macro_chunk,
            micro_chunk=i,
        )
        audio_b64 = _wav_block_to_b64(micro_chunk, wav_info)
        split_percentage = split_index / len(wav_block_buffer) * 100
        logging.info(
            f"Macro-chunk [{download.chunk_info.macro_chunk:03}]: starting micro-chunk "
            f"[{i + 1:03}/~{approx_num_micro_chunks:03}], `{len(micro_chunk)}` "
            f"bytes ({len(audio_b64)} as base64) duration="
            f"{duration_sec:.2f}s, split at {split_percentage:.1f}%."
        )
        if _INJECT_ERRORS:
            utils.random_fail(0.01, "wav_chunking")
        yield seg_info, audio_b64

        start_time_sec = end_time_sec
        wav_block_buffer = wav_block_buffer[split_index:]


def convert_whisper_segments(
    whisper_result: data_types.WhisperResult, chunk_info: data_types.ChunkInfo
) -> Iterable[data_types.Segment]:
    for whisper_segment in whisper_result.segments:
        segment = data_types.Segment(
            start_time_sec=chunk_info.start_time_sec + whisper_segment.start_time_sec,
            end_time_sec=chunk_info.start_time_sec + whisper_segment.end_time_sec,
            text=whisper_segment.text,
            language=whisper_result.language,
            language_code=whisper_result.language_code,
        )
        yield segment


_T = TypeVar("_T")


async def gather(tasks: Iterable[asyncio.Task[_T]]) -> list[_T]:
    return await asyncio.gather(*tasks)
