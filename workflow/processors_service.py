"""
Open questions - safe translation from processors.py to this generated file.

* How to deal with locally defined symbols from `processors.py`?
    - Force all processors in the same file to use the same image/deps and
      add any non `processor`-definition source and imports as common "header" here.
    - Specifically backtrack which symbols are used (recursively) and only add those.
* Locally defined symbols must not reference processors!
* Can processors reference processors?
    - What about class-processors that are not instantiated in `processors.py`?

"""

import pydantic


class WaveObj(pydantic.BaseModel):
    ...


class StitchResult(pydantic.BaseModel):
    ...


class RemoteProcessorBase:
    ...


class Chunking(RemoteProcessorBase):
    def chunking(wav_obj: WaveObj) -> list[str]:
        # Makes call to truss.predict endpoint.
        ...

    def gen_code():
        return ""


class Stitch(RemoteProcessorBase):
    def chunking(wav_obj: WaveObj) -> list[str]:
        # Makes call to truss.predict endpoint.
        ...

    def gen_code():
        imports = """import transformers"""
        init = """self._resource = transformers.blabla()"""
        body = """return self._resource(transcripts)"""


# Deploy all - or at least make them all instances:
with deploy_with_config_overrides(...):
    chunking = Chunking().chunking
    stitch = Stitch().__call__
