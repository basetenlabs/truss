import base64
import random
from functools import partial
from io import BytesIO

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from dalle_mini import DalleBart, DalleBartProcessor
from flax.jax_utils import replicate
from flax.training.common_utils import shard_prng_key
from PIL import Image
from tqdm.notebook import trange
from vqgan_jax.modeling_flax_vqgan import VQModel

# dalle-mini
# can be wandb artifact or 🤗 Hub or local folder or google bucket
DALLE_MODEL = "dalle-mini/dalle-mini/model-mheh9e55:latest"
DALLE_COMMIT_ID = None

# VQGAN model
VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

DTYPE = jnp.float32

# Defaults for generation but can be updated on request
DEFAULT_NUM_PREDICTIONS = 1
GEN_TOP_K = None
GEN_TOP_P = None
TEMP = 0.85
COND_SCALE = 3.0

# we pull resources that require wandb, specifically the DALLE_MODEL
wandb.init(anonymous="must")

# Load dalle-mini
model = DalleBart.from_pretrained(
    DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=DTYPE, abstract_init=True
)
# Load VQGAN
vqgan = VQModel.from_pretrained(VQGAN_REPO, revision=VQGAN_COMMIT_ID)

# Replicate model parameters to any devices we have access to
model._params = replicate(model.params)
vqgan._params = replicate(vqgan.params)


class Singleton(object):
    def __new__(cls, *args, **kwds):
        it = cls.__dict__.get("__it__")
        if it is not None:
            return it
        cls.__it__ = it = object.__new__(cls)
        it.init(*args, **kwds)
        return it

    def init(self, *args, **kwargs):
        pass


# a HuggingFace parallelizer/tokenizer will disable any parallelism if it
# detects that it is in a forked process; so we ensure that there is at most one object per process, and
# initiate it lazily with this singleton pattern
class DalleTokenizer(Singleton):
    def init(self, *args, **kwargs):
        self.tokenizer = DalleBartProcessor.from_pretrained(
            DALLE_MODEL, revision=DALLE_COMMIT_ID
        )


@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
def parallel_generate(
    tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale
):
    """Parallelize inference over all devices that are available"""
    return model.generate(
        **tokenized_prompt,
        prng_key=key,
        params=params,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        condition_scale=condition_scale,
    )


@partial(jax.pmap, axis_name="batch")
def parallel_decode(indices, params):
    """Parallelize image decoding over all devices that availabled"""
    return vqgan.decode_code(indices, params=params)


def tokenize_prompt(prompt: str):
    """Tokenize and replicate the prompt"""
    processor = DalleTokenizer().tokenizer
    tokenized_prompt = processor([prompt])
    return replicate(tokenized_prompt)


class DallEModel(object):
    def load(self):
        pass

    @staticmethod
    def generate_images(
        prompt: str,
        num_predictions: int,
        gen_top_k: int,
        gen_top_p: int,
        temp: float,
        cond_scale: float,
    ):
        tokenized_prompt = tokenize_prompt(prompt)

        # create a random key
        seed = random.randint(0, 2**32 - 1)
        key = jax.random.PRNGKey(seed)

        # generate images
        images = []
        for _ in trange(num_predictions // jax.device_count()):
            # get a new key
            key, subkey = jax.random.split(key)

            # generate images
            encoded_images = parallel_generate(
                tokenized_prompt,
                shard_prng_key(subkey),
                model.params,
                gen_top_k,
                gen_top_p,
                temp,
                cond_scale,
            )

            # remove BOS
            encoded_images = encoded_images.sequences[..., 1:]

            # decode images
            decoded_images = parallel_decode(encoded_images, vqgan.params)
            decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
            for img in decoded_images:
                images.append(Image.fromarray(np.asarray(img * 255, dtype=np.uint8)))

        return images

    def predict_single(self, request):
        prompt = request.get("prompt")
        num_predictions = request.get("num_predictions", DEFAULT_NUM_PREDICTIONS)
        gen_top_k = request.get("top_k", GEN_TOP_K)
        gen_top_p = request.get("top_p", GEN_TOP_P)
        temperature = request.get("temperature", TEMP)
        cond_scale = request.get("condition_scale", COND_SCALE)

        generated_images = self.generate_images(
            prompt,
            num_predictions=num_predictions,
            gen_top_k=gen_top_k,
            gen_top_p=gen_top_p,
            temp=temperature,
            cond_scale=cond_scale,
        )
        single_request_response = []
        for img in generated_images:
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            single_request_response.append({"image_b64": img_b64})

        return single_request_response

    def predict(self, list_of_requests):
        response = []
        for request in list_of_requests:
            response.append(self.predict_single(request))
        return response
