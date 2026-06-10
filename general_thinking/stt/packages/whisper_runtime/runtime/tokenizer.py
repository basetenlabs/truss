import os
from functools import cached_property
from pathlib import Path
from typing import List, Tuple

BASE_PATH = Path(__file__).parent.parent

_TASKS = (
    "transcribe",
    "translate",
)

with open(os.path.join(BASE_PATH, "assets/lang_codes.txt"), "r") as f:
    _LANGUAGE_CODES = [_ for _ in f.read().split("\n") if _]


class Tokenizer:
    def __init__(self, tokenizer):

        self.tokenizer = tokenizer
        self.multilingual = True

        if self.multilingual:
            self.task_to_token_id = {
                task: self.tokenizer.token_to_id(f"<|{task}|>") for task in _TASKS
            }
            self.lang_code_to_token_id = {
                lang: self.tokenizer.token_to_id(f"<|{lang}|>") for lang in _LANGUAGE_CODES
            }
            self.id_to_lang_code = {v: k for k, v in self.lang_code_to_token_id.items()}
        else:
            self.task_to_token_id = None
            self.lang_code_to_token_id = None
            self.id_to_lang_code = None

    @cached_property
    def transcribe(self) -> int:
        return self.tokenizer.token_to_id("<|transcribe|>")

    @cached_property
    def translate(self) -> int:
        return self.tokenizer.token_to_id("<|translate|>")

    @cached_property
    def silent_token(self) -> int:
        return self.encode(" ")[0]

    @cached_property
    def sot(self) -> int:
        return self.tokenizer.token_to_id("<|startoftranscript|>")

    @cached_property
    def sot_lm(self) -> int:
        return self.tokenizer.token_to_id("<|startoflm|>")

    @cached_property
    def sot_prev(self) -> int:
        return self.tokenizer.token_to_id("<|startofprev|>")

    @cached_property
    def no_speech(self) -> int:
        return self.tokenizer.token_to_id("<|no_speech|>")

    @cached_property
    def eot(self) -> int:
        return self.tokenizer.token_to_id("<|endoftext|>")

    @cached_property
    def no_timestamps(self) -> int:
        return self.tokenizer.token_to_id("<|notimestamps|>")

    @property
    def timestamp_begin(self) -> int:
        return self.no_timestamps + 1

    @property
    def all_language_codes(self) -> List[str]:
        return list(self.lang_code_to_token_id.keys()) if self.multilingual else []

    @cached_property
    def all_language_tokens(self) -> Tuple[int]:
        if not self.multilingual:
            return tuple()
        return tuple(self.lang_code_to_token_id.values())

    def sot_sequence(self, task=None, lang=None):
        sequence = [self.sot]

        if self.multilingual:
            sequence.append(self.lang_code_to_token_id[lang])
            sequence.append(self.task_to_token_id[task])

        return sequence

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, tokens, prompt_tokens=[]):
        generated_tokens = tokens[len(prompt_tokens) :]
        text_tokens = [token for token in generated_tokens if token < self.eot and token >= 0]
        return self.tokenizer.decode(text_tokens)

    def decode_batch(self, tokens):
        res = []
        for tk in tokens:
            res.append([token for token in tk if token < self.eot and token >= 0])

        return self.tokenizer.decode_batch(res)

    def split_tokens_on_unicode(self, text, tokens):
        replacement_char = "\ufffd"

        subwords, subword_tokens_list, current_tokens = [], [], []
        unicode_offset, word_finished = 0, False

        for token in tokens:
            current_tokens.append(token)
            decoded = self.decode(current_tokens)

            try:
                replacement_char_index = decoded.index(replacement_char) + unicode_offset
                if (replacement_char_index < len(text)) and (
                    text[replacement_char_index] == replacement_char
                ):
                    word_finished = True
            except ValueError:
                word_finished = True

            if word_finished:
                subwords.append(decoded)
                subword_tokens_list.append(current_tokens)

                current_tokens = []
                word_finished = False
                unicode_offset += len(decoded)

        return subwords, subword_tokens_list

    def split_tokens_on_spaces(self, text, tokens):
        subwords, subword_tokens_list = self.split_tokens_on_unicode(text, tokens)
        words = []
        word_tokens = []

        for subword, subword_tokens in zip(subwords, subword_tokens_list):
            conditions = [
                subword_tokens[0] >= self.eot,  # special
                subword.startswith(" "),  # with_space
                # subword.strip() in string.punctuation, # punctuation
                len(words) == 0,
            ]

            if any(conditions):
                words.append(subword.strip())
                word_tokens.append(subword_tokens)
            else:
                words[-1] = words[-1] + subword
                word_tokens[-1].extend(subword_tokens)

        return words, word_tokens

    def split_to_word_tokens(self, text, tokens, lang_code):
        if lang_code in {"zh", "ja", "th", "lo", "my", "yue"}:
            # These languages don't typically use spaces, so it is difficult to split words
            # without morpheme analysis. Here, we instead split words at any
            # position where the tokens are decoded as valid unicode points
            return self.split_tokens_on_unicode(text, tokens)

        return self.split_tokens_on_spaces(text, tokens)

    def split_to_word_tokens_batch(self, text_batch, tokens_batch, lang_code_batch):
        res = []
        for text, tokens, lang_code in zip(text_batch, tokens_batch, lang_code_batch):
            res.append(self.split_to_word_tokens(text, tokens, lang_code))

        return res

    def token_to_language(self, token: int) -> str:
        return self.id_to_lang_code.get(token, "")
