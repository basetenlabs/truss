"""Universal dataset loader for LLM training - supports multiple formats and streaming."""

import os
from typing import Any, Dict, List, Optional, Union

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict


class UniversalLLMLoader:
    """Loads datasets from HF Hub or local files, normalizing to messages format."""

    def __init__(self, model_name_or_path: Optional[str] = None):
        self.model_name_or_path = model_name_or_path

    def load(
        self,
        path_or_id: str,
        split: Optional[str] = None,
        max_samples: Optional[int] = None,
        streaming: bool = False,
        **kwargs,
    ) -> Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict]:
        """Load dataset, normalizing to messages format."""
        from datasets import load_dataset

        is_local = os.path.exists(path_or_id)
        use_streaming = streaming or (max_samples is not None and not is_local)

        load_kwargs = {k: v for k, v in kwargs.items() if k != "split"}
        load_kwargs["streaming"] = use_streaming

        effective_split = None
        if split is not None and max_samples is not None and "[" not in split and not use_streaming:
            effective_split = f"{split}[:{max_samples}]"
        elif split is not None:
            effective_split = split
        elif max_samples is not None and not use_streaming:
            effective_split = f"train[:{max_samples}]"
        elif use_streaming and max_samples is not None and not is_local:
            effective_split = "train"
        if effective_split is not None:
            load_kwargs["split"] = effective_split

        if is_local:
            ext = os.path.splitext(path_or_id)[-1].replace(".", "")
            loader_type = "json" if ext == "jsonl" else ext
            dataset = load_dataset(loader_type, data_files=path_or_id, **load_kwargs)
        else:
            dataset = load_dataset(path_or_id, **load_kwargs)

        if use_streaming and max_samples is not None:
            dataset = self._take_subset(dataset, max_samples)

        dataset = dataset.map(self._standardize_schema, batched=False)

        if use_streaming and max_samples is not None:
            dataset = self._materialize(dataset)

        return dataset

    def _take_subset(
        self, dataset: Union[DatasetDict, IterableDatasetDict], max_samples: int
    ) -> Union[DatasetDict, IterableDatasetDict]:
        if isinstance(dataset, dict):
            return type(dataset)({k: v.take(max_samples) for k, v in dataset.items()})
        return dataset.take(max_samples)

    def _materialize(
        self, dataset: Union[IterableDataset, IterableDatasetDict]
    ) -> Union[Dataset, DatasetDict]:
        if isinstance(dataset, dict):
            return DatasetDict({k: Dataset.from_list(list(v)) for k, v in dataset.items()})
        return Dataset.from_list(list(dataset))

    def _standardize_schema(self, example: Dict[str, Any]) -> Dict[str, Any]:
        for key in ["messages", "conversations", "chat", "history"]:
            if key in example and isinstance(example[key], list) and len(example[key]) > 0:
                return self._process_conversation_format(example[key])

        instruction_keys = ["instruction", "prompt", "question", "input"]
        output_keys = ["output", "response", "answer", "completion"]
        user_content = next((example.get(k) for k in instruction_keys if k in example), None)
        assistant_content = next((example.get(k) for k in output_keys if k in example), None)
        if user_content is not None and assistant_content is not None:
            return self._process_instruction_output_format(user_content, assistant_content)

        if "text" in example:
            return self._process_raw_text_format(example["text"])

        return example

    def _process_conversation_format(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        first_msg = messages[0]
        if "from" in first_msg:
            return {
                "messages": [
                    {
                        "role": "user" if m["from"] in ["human", "user"] else "assistant",
                        "content": m["value"],
                    }
                    for m in messages
                ]
            }
        return {"messages": messages}

    def _process_instruction_output_format(
        self, user_content: Any, assistant_content: Any
    ) -> Dict[str, Any]:
        return {
            "messages": [
                {"role": "user", "content": str(user_content)},
                {"role": "assistant", "content": str(assistant_content)},
            ]
        }

    def _process_raw_text_format(self, text: str) -> Dict[str, Any]:
        return {"messages": [{"role": "assistant", "content": text}]}
