import pydantic
from truss.workflow import api
from truss.workflow.api import DeploymentConfig

# Images - specify dependencies (and potentially assets - E.g. where is model cache?).
# Images don't need to be overwritten for different deployments (unlike resources).
# For more config-driven approach add helper function to specify image from config file.
chunking_image = api.Image.pip_install("chunking req 2", "chunking req 2")
stitching_image = (
    api.Image.cuda("12.8")
    .pip_requirements_txt("transformers")
    .apt_install("ffmpeg")
    .cache_model("openai/whisper-v3")
    .run_commands("More fancy setyp command")
)

# ComputeResources - might be overwritten by config files, possibly different
# ones to deploy the same workflow different envs.
chunking_compute = api.Compute.cpu(count=4, memory="4Gi")
# Will use sensible default CPU resource in addition to GPU.
stitching_compute = api.Compute.gpu(count=1, platform="T4", memory="10Gi")


class WaveObj(pydantic.BaseModel):
    ...


class StitchResult(pydantic.BaseModel):
    ...


@api.processor(
    name="Chunking",
    image=chunking_image,
    resources=chunking_compute,
)
def chunking(wav_obj: WaveObj) -> list[str]:
    return ["a", "b"]


@api.processor(
    name="Stitch", image=chunking_image, resources=chunking_compute, custom_config=None
)
class Stitch(api.ProcessorBase):
    def __init__(self, custom_config: api.CustomConfigBase):
        super().__init__(custom_config)
        import transformers

        self._resource = transformers.blabla()

    def __call___(self, transcripts: list[dict], param: int = 123) -> dict:
        x = self._resource(transcripts)
        return {}


@api.processor(
    name="Workflow",
    image=chunking_image,
    resources=chunking_compute,
)
def chunking(stitcher: Stitch, wkt_in) -> list[str]:
    urls = chunking(wkt_in.url, batch_size=8, parallel=True)
    transcripts = api.model_by_name("Faster-Whisper-v3", batch=True, parallel=True)(
        urls
    )
    if len(transcripts) > 1:
        return stitcher(transcripts, param=444)
    return transcripts[0]
