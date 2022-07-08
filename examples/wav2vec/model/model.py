from typing import Dict, List

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer


class Wav2VecTransformerModel(object):
    def load(self):
        self.device = 0 if torch.cuda.is_available() else "cpu"
        tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(
            self.device
        )
        self._tokenizer = tokenizer
        self._model = model
        self.ready = True

    def predict(self, input_audio: Dict) -> List:
        with torch.no_grad():
            input_values = self._tokenizer(
                input_audio["inputs"], return_tensors="pt"
            ).input_values
            logits = self._model(input_values.to(self.device)).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self._tokenizer.batch_decode(predicted_ids)[0]
        return [transcription]
