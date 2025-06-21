import requests

from truss.base.constants import (
    HF_MODELS_API_URL,
    OPENAI_COMPATIBLE_TAG,
    OPENAI_NON_COMPATIBLE_TAG,
    TRTLLM_MIN_MEMORY_REQUEST_GI,
)
from truss.base.trt_llm_config import TrussTRTLLMModel
from truss.truss_handle.truss_handle import TrussHandle


def has_no_tags_trt_llm_builder(tr: TrussHandle) -> tuple[str, bool]:
    """
    support transitioning to more openai-compatible schema. @michaelfeil
    # transitioning:
    # 1. Require a tag for all models (today), write to config.yaml client-side, and error.
    # 2. disable new legacy-non-openai pushes server-side.
    # 3. always push the openai-compatible tag client-side.
    # 3. remove the meaning of any tags, including this logic on truss push
    """

    def add_openai_tag(tr: TrussHandle) -> str:
        tr.spec.config.model_metadata["tags"] = [
            OPENAI_COMPATIBLE_TAG
        ] + tr.spec.config.model_metadata.get("tags", [])
        tr.spec.config.write_to_yaml_file(tr.spec.config_path, verbose=False)
        message = f"""TRT-LLM model requires a openai-compatible tag.
Adding the following to your config.yaml file:
```yaml
model_metadata:
  tags:
  - {OPENAI_COMPATIBLE_TAG}
  # for legacy behavior set above line to
  # - {OPENAI_NON_COMPATIBLE_TAG}
```
"""
        return message

    if uses_trt_llm_builder(tr):
        assert tr.spec.config.trt_llm is not None
        trt_llm_config = tr.spec.config.trt_llm.root
        if trt_llm_config.inference_stack == "v2":
            return ("", False)
        if trt_llm_config.build.base_model == TrussTRTLLMModel.ENCODER:
            return ("", False)
        # only briton requires openai-compatible tag, all others don't care about the openai tag
        current_tags = tr.spec.config.model_metadata.get("tags", [])

        if trt_llm_config.build.speculator is not None:
            # spec-dec has no classic backend. OpenAI-mode is forced, regardless of tags.
            if OPENAI_NON_COMPATIBLE_TAG in current_tags:
                return (
                    (
                        f"TRT-LLM models with speculator does not support {OPENAI_NON_COMPATIBLE_TAG} tag. "
                        f"Please migrate to {OPENAI_COMPATIBLE_TAG} tag."
                    ),
                    True,
                )
            elif OPENAI_COMPATIBLE_TAG not in current_tags:
                message = add_openai_tag(tr)
                return (
                    (
                        f"TRT-LLM models with speculator require have model_metadata['tags'] section with ['{OPENAI_COMPATIBLE_TAG}']. (openai-compatible) "
                        f"{message}"
                        f"We have adjusted your config.yaml file to include this tag."
                    ),
                    False,
                )
        elif not any(
            tag in current_tags
            for tag in (OPENAI_COMPATIBLE_TAG, OPENAI_NON_COMPATIBLE_TAG)
        ):
            # inserting new tag client side on truss push
            message = add_openai_tag(tr)
            return (
                (
                    f"TRT-LLM models require a model_metadata['tags'] section with ['{OPENAI_COMPATIBLE_TAG}'] or ['{OPENAI_NON_COMPATIBLE_TAG}']. "
                    f"{message}"
                    f"We have adjusted your config.yaml file to include this tag."
                ),
                False,
            )
        elif OPENAI_NON_COMPATIBLE_TAG in current_tags:
            # legacy non-openai mode is deprecated. We will remove this in the future.
            return (
                (
                    f"TRT-LLM model is using the deprecated model_metadata['tags']={OPENAI_NON_COMPATIBLE_TAG} tag. While this is currently working, please consider migrating to {OPENAI_COMPATIBLE_TAG} tag and request schema."
                ),
                False,
            )

    return ("", False)


def memory_updated_for_trt_llm_builder(tr: TrussHandle) -> bool:
    if uses_trt_llm_builder(tr):
        if tr.spec.memory_in_bytes < TRTLLM_MIN_MEMORY_REQUEST_GI * 1024**3:
            tr.spec.config.resources.memory = f"{TRTLLM_MIN_MEMORY_REQUEST_GI}Gi"
            tr.spec.config.write_to_yaml_file(tr.spec.config_path, verbose=False)
            return True
    return False


def _is_model_public(model_id: str) -> bool:
    """Validate that a huggingface hub model is public.

    The hf hub API will return 401 when trying to access a private or gated model without auth.
    """
    response = requests.get(f"{HF_MODELS_API_URL}/{model_id}")
    return response.status_code == 200


def uses_trt_llm_builder(tr: TrussHandle) -> bool:
    return tr.spec.config.trt_llm is not None
