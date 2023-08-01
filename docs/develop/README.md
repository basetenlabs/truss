---
description: Use Truss to package any ML model for deployment.
---

# Packaging models

In this tutorial, we'll walk through packaging an LLM, WizardLM, as a Truss. By the end of the tutorial, you'll have a model that's ready to deploy.

For more examples, [Truss examples](https://github.com/basetenlabs/truss-examples)

## Creating a Truss

To get started, initialize your Truss with the following command in the CLI:

```
truss init wizardlm-truss
```

This will create the following file structure:

```
wizardlm-truss/     # Truss root directory
  data/         # Stores serialized models/weights/binaries
  model/
    __init__.py
    model.py    # Implements Model class
  packages/     # Stores utility code for model.py
  config.yaml   # Config for model serving environment
  examples.yaml # Invocation examples
```

Most of the development work will happen in `model/model.py` and `config.yaml`.

## Packaging a model

Packaging details vary from model to model, but every model packaging process requires four core steps:

1. Implement model load
2. Implement model invocation
3. Set Python requirements
4. Set hardware requirements

### Implement model load

In `model/model.py`, the first function you'll need to implement is `load()`.

When the model is spun up to receive requests, `load()` is called exactly once and is guaranteed to finish before any inference is attempted.

The exact code you'll need will depend on your model and framework. In this example, model weights for WizardLM are coming from HuggingFace.

Here's the load function in the context of `model/model.py`:

```python
from typing import Any
import torch

# WizardLM uses the same Tokenizer and CausalLM objects as LLaMA
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

class Model:

    def __init__(self, **kwargs) -> None:
        # These values will be set in load()
        self.model = None
        self.tokenizer = None

    def load(self):
        # Load public model from HuggingFace
        base_model = "TheBloke/wizardLM-7B-HF"
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        # Do some WizardLM-specific configuration
        model.config.pad_token_id = tokenizer.pad_token_id = 0
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
        model.half()
        model.eval()

        # Set up the Model object with its model and tokenizer
        self.model = model
        self.tokenizer = tokenizer
```

In this example, our model weights are loaded from Hugging Face.

### Implement model invocation

The other key function in your Truss is `predict()`, which handles model invocation. Here's the `predict` function for WizardLM:

```python
    def predict(self, request) -> Any:
        prompt = request.pop("prompt")
        _output = evaluate(self.model, self.tokenizer, prompt, **request)
        final_output = _output[0].split("### Response:")[1].strip()
        return final_output
```

This function relies on a bit of helper code, adapted from the model card:

```python
def evaluate(
    model,
    tokenizer,
    model_input,
    input=None,
    temperature=1,
    top_p=0.9,
    top_k=40,
    num_beams=1,
    max_new_tokens=2048,
    **kwargs,
):
    prompts = generate_prompt(model_input, input)
    inputs = tokenizer(prompts, return_tensors="pt", max_length=1024, truncation=True, padding=True)
    input_ids = inputs["input_ids"].to("cuda")
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences
    output = tokenizer.batch_decode(s, skip_special_tokens=True)
    return output


def generate_prompt(instruction, input=None):
    return f"""{instruction}

### Response:
"""
```

When the helper code is just a few dozen lines, you may prefer to include it directly in `model/model.py`. If you do, place these functions outside of the `Model` class.

### Set Python and system requirements

The code above relies on some Python imports, and the model itself also has dependencies. `model/model.py` required the following packages:

```python
from typing import Any
import torch

# WizardLM uses the same Tokenizer and CausalLM objects as LLaMA
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
```

To add Python requirements, turn your attention to `config.yaml`. You can use this file to customize a great deal about your packaged model, but right now just set the Python requirements up so the model can run.

For that, find `requirements:` in the config file. In the WizardLM 1.5 example, set it to:

```yaml
requirements:
- torch==2.0.1
- peft==0.3.0
- sentencepiece==0.1.99
- accelerate==0.20.3
- bitsandbytes==0.39.1
- transformers==4.30.2
```

These requirements work just like `requirements.txt` in a Python project, and you can pin versions with `package==1.2.3`.

You can also specify required system packages if needed with `system_packages` in `config.yaml`. WizardLM does not require any system packages.

### Set hardware requirements

Large models like WizardLM require powerful hardware to run invocations. Set your packaged model's hardware requirements in `config.yaml`:

```yaml
resources:
  accelerator: A10G # Type of GPU required
  cpu: "8" # Number of vCPU cores required
  memory: 30Gi # Mibibytes (Mi) or Gibibytes (Gi) of RAM required
  use_gpu: true # If false, set accelerator: null
```

You've successfully packaged a model! Next, [deploy it](../deploy/baseten.md).
