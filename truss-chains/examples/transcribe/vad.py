import logging
import time

import torch


class VAD:
    def __init__(self):
        self._device = "cpu"
        self._vad_model, self._vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            onnx=False,
            trust_repo=True,
        )
        self._get_speech_timestamps = self._vad_utils[0]
        self._save_audio = self._vad_utils[1]
        self._vad_model.to(self._device)

    def get_speech_timestamps(self, audio_bytes: bytes, data):
        """
        Gets the speech timestamps for the given audio bytes.
        """
        t0 = time.time()
        audio = (
            torch.frombuffer(audio_bytes, dtype=torch.int16).to(torch.float32) / 32768.0
        )
        result = self._get_speech_timestamps(
            audio,
            self._vad_model,
            sampling_rate=16000,
            return_seconds=True,
            threshold=0.5,
            min_silence_duration_ms=500,
        )
        # self._save_audio("test.wav", audio, 16000)
        logging.info(f"VAD took {time.time() - t0} seconds")
        return result
