import logging
import math
import time
import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
import torchaudio

languages = ["ru", "en", "de", "es"]

logger = logging.getLogger(__name__)
@dataclass
class BatchSettings:
    lookback_window_secs: float = 4
    max_batch_size: int = 400
    min_section_length: int = 32


class OnnxWrapper:

    def __init__(self, path, force_onnx_cpu=False):
        import numpy as np

        global np
        import onnxruntime

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        if (
            force_onnx_cpu
            and "CPUExecutionProvider" in onnxruntime.get_available_providers()
        ):
            self.session = onnxruntime.InferenceSession(
                path, providers=["CPUExecutionProvider"], sess_options=opts
            )
            self._device = "cpu"
        else:
            self.session = onnxruntime.InferenceSession(
                path, providers=["CUDAExecutionProvider"], sess_options=opts
            )
            self._device = "cuda"
        
        self._state = {}
        self._context = {}
        self._last_sr = {}
        self._last_batch_size = {}

        self.sample_rates = [8000, 16000]

    def _validate_input(self, x, sr: int):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() > 2:
            raise ValueError(f"Too many dimensions for input audio chunk {x.dim()}")

        if sr != 16000 and (sr % 16000 == 0):
            step = sr // 16000
            x = x[:, ::step]
            sr = 16000

        if sr not in self.sample_rates:
            raise ValueError(
                f"Supported sampling rates: {self.sample_rates} (or multiply of 16000)"
            )
        if sr / x.shape[1] > 31.25:
            raise ValueError("Input audio chunk is too short")

        return x, sr

    def reset_states(self, id, batch_size=1):
        self._state[id] = torch.zeros((2, batch_size, 128)).float().to(self._device)
        self._context[id] = torch.zeros(0).to(self._device)
        self._last_sr[id] = 0
        self._last_batch_size[id] = 0
    
    def delete_states(self, id):
        del self._state[id]
        del self._context[id]
        del self._last_sr[id]
        del self._last_batch_size[id]

    def __call__(self, id, x, sr: int):

        assert id in self._state, f"State for id {id} not found in VAD OnnxWrapper"

        x, sr = self._validate_input(x, sr)
        num_samples = 512 if sr == 16000 else 256

        if x.shape[-1] != num_samples:
            raise ValueError(
                f"Provided number of samples is {x.shape[-1]} (Supported values: 256 for 8000 sample rate, 512 for 16000)"
            )

        batch_size = x.shape[0]
        context_size = 64 if sr == 16000 else 32

        if not self._last_batch_size[id]:
            self.reset_states(id, batch_size)
        if (self._last_sr[id]) and (self._last_sr[id] != sr):
            self.reset_states(id, batch_size)
        if (self._last_batch_size[id]) and (self._last_batch_size[id] != batch_size):
            self.reset_states(id, batch_size)

        if not len(self._context[id]):
            self._context[id] = torch.zeros(batch_size, context_size).to(self._device)

        x = torch.cat([self._context[id], x], dim=1)
        if sr in [8000, 16000]:
            ort_inputs = {
                "input": x.cpu().numpy(),
                "state": self._state[id].cpu().numpy(),
                "sr": np.array(sr, dtype="int64"),
            }
            ort_outs = self.session.run(None, ort_inputs)
            out, state = ort_outs
            self._state[id] = torch.from_numpy(state)
        else:
            raise ValueError()

        self._context[id] = x[..., -context_size:]
        self._last_sr[id] = sr
        self._last_batch_size[id] = batch_size

        out = torch.from_numpy(out)
        return out

    # TODO: Not tested after id update yet
    def audio_forward(self, id: str, x, sr: int):
        outs = []
        x, sr = self._validate_input(x, sr)
        self.reset_states(id)
        num_samples = 512 if sr == 16000 else 256

        if x.shape[1] % num_samples:
            pad_num = num_samples - (x.shape[1] % num_samples)
            x = torch.nn.functional.pad(x, (0, pad_num), "constant", value=0.0)

        for i in range(0, x.shape[1], num_samples):
            wavs_batch = x[:, i : i + num_samples]
            out_chunk = self.__call__(id, wavs_batch, sr)
            outs.append(out_chunk)

        stacked = torch.cat(outs, dim=1)
        return stacked.cpu()


class Validator:
    def __init__(self, url, force_onnx_cpu):
        self.onnx = True if url.endswith(".onnx") else False
        torch.hub.download_url_to_file(url, "inf.model")
        if self.onnx:
            import onnxruntime

            if (
                force_onnx_cpu
                and "CPUExecutionProvider" in onnxruntime.get_available_providers()
            ):
                self.model = onnxruntime.InferenceSession(
                    "inf.model", providers=["CPUExecutionProvider"]
                )
            else:
                self.model = onnxruntime.InferenceSession("inf.model")
        else:
            self.model = init_jit_model(model_path="inf.model")

    def __call__(self, inputs: torch.Tensor):
        with torch.no_grad():
            if self.onnx:
                ort_inputs = {"input": inputs.cpu().numpy()}
                outs = self.model.run(None, ort_inputs)
                outs = [torch.Tensor(x) for x in outs]
            else:
                outs = self.model(inputs)

        return outs


def read_audio_ffmpeg(path: str, sampling_rate: int = 16000):
    # Load the audio file
    start_time = time.time()
    wav, sr = torchaudio.load(path, backend="ffmpeg")
    end_time_1 = time.time()
    print(f"torchaudio.load: {end_time_1 - start_time} seconds")

    # Convert to mono if necessary
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    end_time_2 = time.time()
    print(f"wav.mean: {end_time_2 - end_time_1} seconds")

    # Resample if necessary
    if sr != sampling_rate:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=sampling_rate)
    end_time_3 = time.time()
    print(f"torchaudio.functional.resample: {end_time_3 - end_time_2} seconds")

    return wav.squeeze(0)


def read_audio(path: str, sampling_rate: int = 16000):
    list_backends = torchaudio.list_audio_backends()

    assert (
        len(list_backends) > 0
    ), "The list of available backends is empty, please install backend manually. \
                                    \n Recommendations: \n \tSox (UNIX OS) \n \tSoundfile (Windows OS, UNIX OS) \n \tffmpeg (Windows OS, UNIX OS)"

    try:
        effects = [["channels", "1"], ["rate", str(sampling_rate)]]

        print(f"Trying to load {path} with torchaudio.sox_effects")
        start_time = time.time()

        wav, sr = torchaudio.sox_effects.apply_effects_file(path, effects=effects)
        end_time = time.time()
        print(
            f"Loaded {path} with torchaudio.sox_effects in {end_time - start_time} seconds"
        )
    except:
        wav, sr = torchaudio.load(path)

        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != sampling_rate:
            transform = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=sampling_rate
            )
            wav = transform(wav)
            sr = sampling_rate

    return wav.squeeze(0)


def save_audio(path: str, tensor: torch.Tensor, sampling_rate: int = 16000):
    torchaudio.save(path, tensor.unsqueeze(0), sampling_rate, bits_per_sample=16)


def init_jit_model(model_path: str, device=torch.device("cpu")):
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model


def make_visualization(probs, step):
    import pandas as pd

    pd.DataFrame({"probs": probs}, index=[x * step for x in range(len(probs))]).plot(
        figsize=(16, 8),
        kind="area",
        ylim=[0, 1.05],
        xlim=[0, len(probs) * step],
        xlabel="seconds",
        ylabel="speech probability",
        colormap="tab20",
    )

class SileroVAD:
    def __init__(self, model, device):
        '''
        SileroVAD class is a wrapper around either the JIT or ONNX model.

        Args:
            model: The model to wrap (either the JIT or ONNX model)
            device: The device to run the model on
        '''

        assert isinstance(model, (torch.jit.ScriptModule, OnnxWrapper)), "Model must be a JIT or ONNX model"
        self.is_jit = isinstance(model, torch.jit.ScriptModule)
        self.model = model
        self.device = device

    def delete_states(self, id):
        if self.is_jit:
            return

        self.model.delete_states(id)
    
    def reset_states(self, id):
        if self.is_jit:
            self.model.reset_states()
        else:
            self.model.reset_states(id)
    
    def __call__(self, id, x, sr: int):
        if self.is_jit:
            return self.model(x, sr)
        else:
            return self.model(id, x, sr)
    
    def get_num_states(self):
        if self.is_jit:
            return 1
        else:
            return len(self.model._state)


def detect_speeches_from_probs(
    speech_probs,
    threshold,
    neg_threshold,
    min_silence_samples,
    min_silence_samples_at_max_speech,
    min_speech_samples,
    max_speech_samples,
    window_size_samples,
    audio_length_samples,
    sampling_rate,
    speech_pad_samples,
    return_seconds,
    step,
    speech_probs_offset=0,  # offset in terms of windows
):
    triggered = False
    speeches = []
    current_speech = {}
    temp_end = 0
    prev_end = next_start = 0

    num_probs = len(speech_probs)

    for idx, speech_prob in enumerate(speech_probs):
        i = idx  # index within speech_probs
        sample_idx = (speech_probs_offset + i) * window_size_samples

        if (speech_prob >= threshold) and temp_end:
            temp_end = 0
            if next_start < prev_end:
                next_start = sample_idx

        if (speech_prob >= threshold) and not triggered:
            triggered = True
            current_speech["start"] = sample_idx
            continue

        if triggered and (sample_idx - current_speech["start"] > max_speech_samples):
            if prev_end:
                current_speech["end"] = prev_end
                speeches.append(current_speech)
                current_speech = {}
                if next_start < prev_end:
                    triggered = False
                else:
                    current_speech["start"] = next_start
                prev_end = next_start = temp_end = 0
            else:
                current_speech["end"] = sample_idx
                speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                continue

        if (speech_prob < neg_threshold) and triggered:
            if not temp_end:
                temp_end = sample_idx
            if (sample_idx - temp_end) > min_silence_samples_at_max_speech:
                prev_end = temp_end
            if (sample_idx - temp_end) < min_silence_samples:
                continue
            else:
                current_speech["end"] = temp_end
                if (
                    current_speech["end"] - current_speech["start"]
                ) > min_speech_samples:
                    speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                continue

    if (
        current_speech
        and (audio_length_samples - current_speech["start"]) > min_speech_samples
    ):
        current_speech["end"] = audio_length_samples
        speeches.append(current_speech)

    # Apply padding and adjust start and end times
    for i, speech in enumerate(speeches):
        if i == 0:
            speech["start"] = int(max(0, speech["start"] - speech_pad_samples))
        if i != len(speeches) - 1:
            silence_duration = speeches[i + 1]["start"] - speech["end"]
            if silence_duration < 2 * speech_pad_samples:
                speech["end"] += int(silence_duration // 2)
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - silence_duration // 2)
                )
            else:
                speech["end"] = int(
                    min(audio_length_samples, speech["end"] + speech_pad_samples)
                )
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - speech_pad_samples)
                )
        else:
            speech["end"] = int(
                min(audio_length_samples, speech["end"] + speech_pad_samples)
            )

    if return_seconds:
        for speech_dict in speeches:
            speech_dict["start"] = round(speech_dict["start"] / sampling_rate, 1)
            speech_dict["end"] = round(speech_dict["end"] / sampling_rate, 1)
    elif step > 1:
        for speech_dict in speeches:
            speech_dict["start"] *= step
            speech_dict["end"] *= step

    return speeches


@torch.no_grad()
def get_speech_timestamps(
    audio: torch.Tensor,
    model: SileroVAD,
    threshold: float = 0.5,
    sampling_rate: int = 16000,
    min_speech_duration_ms: int = 250,
    max_speech_duration_s: float = float("inf"),
    retry_max_speech_duration_s: float = float("inf"),
    min_silence_duration_ms: int = 100,
    retry_min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
    return_seconds: bool = False,
    visualize_probs: bool = False,
    progress_tracking_callback: Callable[[float], None] = None,
    window_size_samples: int = 512,
    batch_settings: Optional[BatchSettings] = None,
):
    logger.debug(f"Calling get_speech_timestamps with {threshold=} {sampling_rate=} {min_speech_duration_ms=} {max_speech_duration_s=} {retry_max_speech_duration_s=} {min_silence_duration_ms=} {retry_min_silence_duration_ms=} {speech_pad_ms=} {return_seconds=} {visualize_probs=} {progress_tracking_callback=} {window_size_samples=} {batch_settings=}")
    if not torch.is_tensor(audio):
        try:
            audio = torch.Tensor(audio)
        except:
            raise TypeError("Audio cannot be casted to tensor. Cast it manually")

    if len(audio.shape) > 1:
        for i in range(len(audio.shape)):  # trying to squeeze empty dimensions
            audio = audio.squeeze(0)
        if len(audio.shape) > 1:
            raise ValueError(
                "More than one dimension in audio. Are you trying to process audio with 2 channels?"
            )

    if sampling_rate > 16000 and (sampling_rate % 16000 == 0):
        step = sampling_rate // 16000
        sampling_rate = 16000
        audio = audio[::step]
        warnings.warn(
            "Sampling rate is a multiply of 16000, casting to 16000 manually!"
        )
    else:
        step = 1

    if sampling_rate not in [8000, 16000]:
        raise ValueError(
            "Currently silero VAD models support 8000 and 16000 (or multiply of 16000) sample rates"
        )

    window_size_samples = 512 if sampling_rate == 16000 else 256

    # Reset model states
    model_id = "temp"

    model.reset_states(model_id)
    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000
    max_speech_samples = (
        sampling_rate * max_speech_duration_s
        - window_size_samples
        - 2 * speech_pad_samples
    )
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    min_silence_samples_at_max_speech = sampling_rate * 98 / 1000

    audio_length_samples = len(audio)

    # Calculate speech probabilities
    speech_probs = []
    if batch_settings is None:
        for current_start_sample in range(0, audio_length_samples, window_size_samples):
            chunk = audio[
                current_start_sample : current_start_sample + window_size_samples
            ]
            if len(chunk) < window_size_samples:
                chunk = torch.nn.functional.pad(
                    chunk, (0, int(window_size_samples - len(chunk)))
                )
            speech_prob = model(model_id, chunk, sampling_rate).item()
            speech_probs.append(speech_prob)
            # Calculate progress and send it to callback function
            progress = current_start_sample + window_size_samples
            if progress > audio_length_samples:
                progress = audio_length_samples
            progress_percent = (progress / audio_length_samples) * 100
            if progress_tracking_callback:
                progress_tracking_callback(progress_percent)
    else:
        # Batch prediction
        speech_probs = _batch_predict(
            model,
            model_id,
            audio,
            window_size_samples,
            batch_settings,
            sampling_rate,
        )

    # First, detect speeches with original parameters
    neg_threshold = threshold - 0.15

    speeches = detect_speeches_from_probs(
        speech_probs,
        threshold,
        neg_threshold,
        min_silence_samples,
        min_silence_samples_at_max_speech,
        min_speech_samples,
        max_speech_samples,
        window_size_samples,
        audio_length_samples,
        sampling_rate,
        speech_pad_samples,
        return_seconds=False,  # We'll handle conversion later
        step=1,
        speech_probs_offset=0,
    )

    # Now, for each speech in speeches, detect more chunked speeches
    # Use a shorter silence tolerance (e.g., 800 windows)
    more_chunked_speeches_list = []
    retry_min_silence_samples = sampling_rate * retry_min_silence_duration_ms / 1000
    retry_max_speech_samples = (
        sampling_rate * retry_max_speech_duration_s
        - window_size_samples
        - 2 * speech_pad_samples
    )

    # For each speech segment in speeches
    for speech in speeches:
        start_sample = speech["start"]
        end_sample = speech["end"]
        # Compute start and end indices in speech_probs
        start_idx = start_sample // window_size_samples
        end_idx = (
            end_sample + window_size_samples - 1
        ) // window_size_samples  # Ensure coverage

        # Extract the portion of speech_probs corresponding to this speech segment
        speech_probs_segment = speech_probs[start_idx:end_idx]

        # Detect speeches in this segment with modified parameters
        chunked_speeches = detect_speeches_from_probs(
            speech_probs_segment,
            threshold,
            neg_threshold,
            min_silence_samples=retry_min_silence_samples,
            min_silence_samples_at_max_speech=min_silence_samples_at_max_speech,
            min_speech_samples=min_speech_samples,
            max_speech_samples=retry_max_speech_samples,
            window_size_samples=window_size_samples,
            audio_length_samples=end_sample,
            sampling_rate=sampling_rate,
            speech_pad_samples=speech_pad_samples,
            return_seconds=False,  # We'll handle conversion later
            step=1,
            speech_probs_offset=start_idx,
        )

        more_chunked_speeches_list.append(chunked_speeches)

    # Now, handle return_seconds and step adjustments for speeches and more_chunked_speeches_list
    if return_seconds:
        for speech_dict in speeches:
            speech_dict["start"] = round(speech_dict["start"] / sampling_rate, 1)
            speech_dict["end"] = round(speech_dict["end"] / sampling_rate, 1)
        for chunked_speeches in more_chunked_speeches_list:
            for speech_dict in chunked_speeches:
                speech_dict["start"] = round(speech_dict["start"] / sampling_rate, 1)
                speech_dict["end"] = round(speech_dict["end"] / sampling_rate, 1)
    elif step > 1:
        for speech_dict in speeches:
            speech_dict["start"] *= step
            speech_dict["end"] *= step
        for chunked_speeches in more_chunked_speeches_list:
            for speech_dict in chunked_speeches:
                speech_dict["start"] *= step
                speech_dict["end"] *= step

    if visualize_probs:
        make_visualization(speech_probs, window_size_samples / sampling_rate)

    model.delete_states(model_id)

    return speeches, more_chunked_speeches_list

# TODO: Not tested after id update yet
class VADIterator:
    def __init__(
        self,
        model: SileroVAD,
        id: str,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
    ):
        """
        Class for stream imitation

        Parameters
        ----------

        id: str
            ID of the stream. Used for ONNX VAD model state management.

        model: preloaded .jit/.onnx silero VAD model (SileroVAD class)

        threshold: float (default - 0.5)
            Speech threshold. Silero VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value are considered as SPEECH.
            It is better to tune this parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.

        sampling_rate: int (default - 16000)
            Currently silero VAD models support 8000 and 16000 sample rates

        min_silence_duration_ms: int (default - 100 milliseconds)
            In the end of each speech chunk wait for min_silence_duration_ms before separating it

        speech_pad_ms: int (default - 30 milliseconds)
            Final speech chunks are padded by speech_pad_ms each side
        """

        self.model = model
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.id = id

        if sampling_rate not in [8000, 16000]:
            raise ValueError(
                "VADIterator does not support sampling rates other than [8000, 16000]"
            )

        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self.speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        self.reset_states()

    def reset_states(self):
        self.model.reset_states(self.id)
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    @torch.no_grad()
    def __call__(self, x, return_seconds=False):
        """
        x: torch.Tensor
            audio chunk (see examples in repo)

        return_seconds: bool (default - False)
            whether return timestamps in seconds (default - samples)
        """

        if not torch.is_tensor(x):
            try:
                x = torch.Tensor(x)
            except:
                raise TypeError("Audio cannot be casted to tensor. Cast it manually")

        window_size_samples = len(x[0]) if x.dim() == 2 else len(x)
        self.current_sample += window_size_samples

        speech_prob = self.model(self.id, x, self.sampling_rate).item()

        if (speech_prob >= self.threshold) and self.temp_end:
            self.temp_end = 0

        if (speech_prob >= self.threshold) and not self.triggered:
            self.triggered = True
            speech_start = max(
                0, self.current_sample - self.speech_pad_samples - window_size_samples
            )
            return {
                "start": (
                    int(speech_start)
                    if not return_seconds
                    else round(speech_start / self.sampling_rate, 1)
                )
            }

        if (speech_prob < self.threshold - 0.15) and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return None
            else:
                speech_end = (
                    self.temp_end + self.speech_pad_samples - window_size_samples
                )
                self.temp_end = 0
                self.triggered = False
                return {
                    "end": (
                        int(speech_end)
                        if not return_seconds
                        else round(speech_end / self.sampling_rate, 1)
                    )
                }

        return None


class StreamingVADIterator:
    def __init__(
        self,
        model: SileroVAD,
        id: str,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_silence_duration_ms: int = 300,
        speech_pad_ms: int = 0,
    ):
        """
        Class for streaming VAD on small audio chunks. Returns final segments once enough
        silence has been detected to close the utterance.

        Parameters
        ----------
        id: str
            ID of the stream. Used for .onnx VAD model state management.

        model: preloaded .jit or .onnx silero VAD model

        threshold: float (default - 0.5)
            Speech threshold. Silero VAD outputs speech probabilities for each audio chunk,
            probabilities ABOVE this value are considered SPEECH.

        sampling_rate: int (default - 16000)
            Currently silero VAD models support 8000 or 16000 sample rates only.

        min_silence_duration_ms: int (default - 100)
            The minimum duration of silence (in ms) needed to decide "speech ended."

        speech_pad_ms: int (default - 30)
            Amount of padding (in ms) to add on both sides of the detected speech segment.
        """

        self.model = model
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.id = id

        if sampling_rate not in [8000, 16000]:
            raise ValueError(
                "StreamingVADIterator does not support sampling rates other than [8000, 16000]"
            )

        # Convert ms to samples
        self.min_silence_samples = int(sampling_rate * min_silence_duration_ms / 1000)
        self.speech_pad_samples = int(sampling_rate * speech_pad_ms / 1000)

        self.reset_states()

    def reset_states(self):
        """
        Reset the internal states of the VAD model and the stream state.
        Call this whenever you start a fresh audio stream.
        """
        self.model.reset_states(self.id)
        self.triggered = False      # Are we currently "in speech"?
        self.temp_end = 0          # Track when we first go below threshold
        self.current_sample = 0    # Cumulative sample count
        self.speech_start = None   # Where the speech segment began

    @torch.no_grad()
    def __call__(self, x, return_seconds=False):
        """
        Process a single chunk of audio (30-50 ms, or any short frame).

        Parameters
        ----------
        x : torch.Tensor
            audio chunk of shape [N] or [1, N].

        return_seconds : bool (default: False)
            whether to return timestamps in seconds (True) or in samples (False).

        Returns
        -------
        dict or None
            - None if no new final segment was found in this chunk
            - A dictionary { "start": ..., "end": ... } once enough silence is detected
              to finalize the utterance. 'start' and 'end' will be either sample indices
              (ints) or seconds (floats).
        """
        # Ensure x is a Tensor
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        # Figure out how many samples are in this chunk
        window_size_samples = len(x[0]) if x.dim() == 2 else len(x)
        self.current_sample += window_size_samples

        # Get speech probability for this chunk
        speech_prob = self.model(self.id, x, self.sampling_rate).item()

        # If we detect speech and we were in silence before, mark the start (internally).
        if (speech_prob >= self.threshold) and not self.triggered:
            self.triggered = True
            # Mark the start slightly earlier for "speech_pad_ms"
            self.speech_start = max(
                0,
                self.current_sample - self.speech_pad_samples - window_size_samples
            )

        # If we are in speech and the probability goes below threshold, track the "temp_end".
        if self.triggered and (speech_prob < self.threshold - 0.15):
            if not self.temp_end:
                self.temp_end = self.current_sample

            # If we've been below threshold long enough => finalize the utterance
            if (self.current_sample - self.temp_end) >= self.min_silence_samples:
                # The final end = first silent sample plus the left padding
                speech_end = self.temp_end + self.speech_pad_samples - window_size_samples

                # Build the result object
                if return_seconds:
                    start_time = round(self.speech_start / self.sampling_rate, 3)
                    end_time = round(speech_end / self.sampling_rate, 3)
                else:
                    start_time = int(self.speech_start)
                    end_time = int(speech_end)

                result = { "start": start_time, "end": end_time }

                # Reset state so we can detect the next utterance
                self.triggered = False
                self.temp_end = 0
                self.speech_start = None

                return result

        # If currently in speech but we see speech_prob >= threshold again,
        # reset temp_end because the silence was interrupted.
        if self.triggered and (speech_prob >= self.threshold):
            self.temp_end = 0

        # If we reach here, there's no new finalized segment
        return None
     
    def __del__(self):
        self.model.delete_states(self.id)
        logger.debug(f"🟢 VAD iterator deleted for id {self.id}. There are {self.model.get_num_states()} states left.")

def collect_chunks(tss: List[dict], wav: torch.Tensor):
    chunks = []
    for i in tss:
        chunks.append(wav[i["start"] : i["end"]])
    return torch.cat(chunks)


def drop_chunks(tss: List[dict], wav: torch.Tensor):
    chunks = []
    cur_start = 0
    for i in tss:
        chunks.append((wav[cur_start : i["start"]]))
        cur_start = i["end"]
    return torch.cat(chunks)

def _batch_predict(
    model, model_id, audio, window_size_samples, batch_settings: BatchSettings, sampling_rate: int
):
    device = model.device
    audio = audio.to(device)
    audio_length_samples = len(audio)
    num_chunks = int(math.ceil(audio_length_samples / window_size_samples))
    to_pad_for_chunking = num_chunks * window_size_samples - audio_length_samples
    padded_audio = torch.nn.functional.pad(
        audio, (0, to_pad_for_chunking), "constant", 0
    )
    chunks = padded_audio.view(num_chunks, window_size_samples)

    # In number of chunks
    section_length = _calc_section_length(
        num_chunks,
        batch_settings.max_batch_size,
        batch_settings.min_section_length,
    )
    num_sections = int(math.ceil(num_chunks / section_length))

    # In number of chunks
    lookback_window = int(
        batch_settings.lookback_window_secs * sampling_rate / window_size_samples
    )
    chunks = F.pad(chunks, (0, 0, 0, lookback_window + section_length), "constant", 0)
    chunk_probabilities = torch.zeros(
        num_sections * section_length + lookback_window
    ).to(device)
    for step in range(lookback_window + section_length):
        step_index_range = slice(
            step, num_sections * section_length + step, section_length
        )
        chunk_batch = chunks[step_index_range]
        model_probs = model(model_id, chunk_batch, sampling_rate)
        # skip first LOOKBACK_WINDOW steps, because we don't have enough context.
        if step < lookback_window:
            chunk_probabilities[step] = model_probs[0].item()
        else:
            chunk_probabilities[step_index_range] = model_probs.view(-1)
    return chunk_probabilities.to(torch.device("cpu")).tolist()[:num_chunks]


def _calc_section_length(
    num_chunks: int, max_batch_size: int, min_section_length: int = 32
):
    min_section_length = 32
    section_length_based_on_max_batch_size = int(math.ceil(num_chunks / max_batch_size))
    return max(min_section_length, section_length_based_on_max_batch_size)
