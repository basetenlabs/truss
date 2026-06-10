from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class Chunker:
    """
    A class to chunk audio signals into speech segments using a Voice Activity Detection (VAD) model.
    """

    def __init__(
        self,
        vad_model: Optional[Any] = None,
        device: Optional[str] = None,
        frame_size: float = 0.02,
        min_seg_len: float = 0.08,
        max_seg_len: float = 29.0,
        max_silent_region: float = 0.6,
        padding: float = 0.2,
        eos_thresh: float = 0.3,
        bos_thresh: float = 0.3,
        cut_factor: int = 2,
        sampling_rate: int = 16000,
    ):
        """
        Initialize the Chunker with the given parameters.

        Args:
            vad_model: Optional[Any]: The VAD model to use. If None, a default VAD model will be created.
            device: Optional[str]: The device to run the VAD model on.
            frame_size: float: The size of each frame in seconds.
            min_seg_len: float: The minimum length of a speech segment in seconds.
            max_seg_len: float: The maximum length of a speech segment in seconds.
            max_silent_region: float: The maximum length of a silent region in seconds.
            padding: float: The padding to add to each segment in seconds.
            eos_thresh: float: The end-of-speech threshold.
            bos_thresh: float: The beginning-of-speech threshold.
            cut_factor: int: The factor to determine the cut index.
            sampling_rate: int: The sampling rate of the audio signal.
        """
        if vad_model is None:
            from .vad import VAD

            vad_model = VAD(device=device)

        self.vad_model = vad_model
        self.sampling_rate = sampling_rate
        self.padding = padding
        self.frame_size = frame_size
        self.min_seg_len = min_seg_len
        self.max_seg_len = max_seg_len
        self.max_silent_region = max_silent_region
        self.eos_thresh = eos_thresh
        self.bos_thresh = bos_thresh
        self.cut_factor = cut_factor
        self.cut_idx = int(self.max_seg_len / (self.cut_factor * self.frame_size))
        self.max_idx_in_seg = self.cut_factor * self.cut_idx

    def update_params(self, params: Dict[str, Any]) -> None:
        """
        Update the parameters of the Chunker.

        Args:
            params: Dict[str, Any]: A dictionary of parameters to update.
        """
        for key, value in params.items():
            setattr(self, key, value)

        self.cut_idx = int(self.max_seg_len / (self.cut_factor * self.frame_size))
        self.max_idx_in_seg = self.cut_factor * self.cut_idx

    def update_vad_model_params(self, params: Dict[str, Any]) -> None:
        """
        Update the parameters of the VAD model.

        Args:
            params: Dict[str, Any]: A dictionary of parameters to update.
        """
        self.vad_model.update_params(params=params)

    def okay_to_merge(
        self,
        speech_probs: np.ndarray,
        last_seg: Dict[str, int],
        curr_seg: Dict[str, int],
    ) -> bool:
        """
        Check if two speech segments can be merged based on maximum silent region and maximum segment length.

        Args:
            speech_probs: np.ndarray: The speech probabilities.
            last_seg: Dict[str, int]: The last speech segment.
            curr_seg: Dict[str, int]: The current speech segment.

        Returns:
            bool: True if the segments can be merged, False otherwise.
        """
        conditions = [
            (speech_probs[curr_seg["start"]][1] - speech_probs[last_seg["end"]][2])
            < self.max_silent_region,
            (speech_probs[curr_seg["end"]][2] - speech_probs[last_seg["start"]][1])
            <= self.max_seg_len,
        ]

        return all(conditions)

    def get_speech_segments(self, speech_probs: np.ndarray) -> List[Tuple[float, float]]:
        """
        Get the speech segments from the speech probabilities.

        Args:
            speech_probs: np.ndarray: The speech probabilities.
            return: List[Tuple[float, float]]: A list of tuples containing the start and end times of the speech segments.
        """
        speech_flag, start_idx = False, 0
        speech_segments = []
        for idx, (speech_prob, st, et) in enumerate(speech_probs):
            if speech_flag:
                if speech_prob < self.eos_thresh:
                    speech_flag = False
                    curr_seg = {"start": start_idx, "end": idx - 1}

                    if len(speech_segments) and self.okay_to_merge(
                        speech_probs, speech_segments[-1], curr_seg
                    ):
                        speech_segments[-1]["end"] = curr_seg["end"]
                    else:
                        speech_segments.append(curr_seg)

            elif speech_prob >= self.bos_thresh:
                speech_flag = True
                start_idx = idx

        if speech_flag:
            curr_seg = {"start": start_idx, "end": len(speech_probs) - 1}

            if len(speech_segments) and self.okay_to_merge(
                speech_probs, speech_segments[-1], curr_seg
            ):
                speech_segments[-1]["end"] = curr_seg["end"]
            else:
                speech_segments.append(curr_seg)

        speech_segments = [
            _
            for _ in speech_segments
            if (speech_probs[_["end"]][2] - speech_probs[_["start"]][1]) > self.min_seg_len
        ]

        start_ends = []
        for _ in speech_segments:
            first_idx = len(start_ends)
            start_idx, end_idx = _["start"], _["end"]
            while (end_idx - start_idx) > self.max_idx_in_seg:
                _start_idx = int(start_idx + self.cut_idx)
                _end_idx = int(min(end_idx, start_idx + self.max_idx_in_seg))

                new_end_idx = _start_idx + np.argmin(speech_probs[_start_idx:_end_idx, 0])
                start_ends.append([speech_probs[start_idx][1], speech_probs[new_end_idx][2]])
                start_idx = new_end_idx + 1

            start_ends.append([speech_probs[start_idx][1], speech_probs[end_idx][2] + self.padding])
            start_ends[first_idx][0] = start_ends[first_idx][0] - self.padding

        return start_ends

    def __call__(
        self,
        audio_signal: Optional[np.ndarray] = None,
    ) -> Tuple[List[Tuple[float, float]], np.ndarray]:
        """
        Process the input audio file or audio signal to get speech segments.

        Args:
            input_file: Optional[str]: The path to the input audio file.
            audio_signal: Optional[np.ndarray]: The audio signal as a numpy array.

        Returns:
            Tuple[List[Tuple[float, float]], np.ndarray]: A tuple containing the list of speech segments and the audio signal.
        """
        audio_duration = len(audio_signal) / self.sampling_rate

        speech_probs = self.vad_model(audio_signal)
        start_ends = self.get_speech_segments(speech_probs)

        if len(start_ends) == 0:
            start_ends = [[0.0, self.max_seg_len]]  # Quick fix for silent audio.

        start_ends[0][0] = max(0.0, start_ends[0][0])  # fix edges
        start_ends[-1][1] = min(audio_duration, start_ends[-1][1])  # fix edges

        return start_ends, audio_signal
