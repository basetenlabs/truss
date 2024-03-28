import asyncio

import chunking
import slay
from slay import stub


class DeployedWhisper(stub.StubBase):
    async def run(self, index: int, audio_b64: str) -> tuple[int, str]:
        response_json = await self._remote.predict_async(
            json_paylod={"audio": audio_b64}
        )
        text_result = response_json["text"]
        return index, text_result


class ChunkedTranscribe(slay.ProcessorBase):

    default_config = slay.Config(
        image=slay.Image()
        .apt_requirements(["ffmpeg"])
        .pip_requirements_file(slay.make_abs_path_here("requirements.txt"))
    )

    def __init__(
        self,
        context: slay.Context = slay.provide_context(),
    ) -> None:
        super().__init__(context)
        self._whisper = DeployedWhisper(
            "https://model-5woz91z3.api.baseten.co/production",
            context.get_baseten_api_key(),
        )

    async def run(self, media_url: str) -> str:
        tasks = []
        for index, audio_b64 in chunking.download_and_generate_chunks(media_url):
            tasks.append(asyncio.ensure_future(self._whisper.run(index, audio_b64)))
        results = await asyncio.gather(*tasks)
        results.sort(key=lambda x: x[0])  # Sort by index.
        merged_text = " ".join(text for _, text in results)
        return merged_text


if __name__ == "__main__":
    # import os
    # TEST_URL = "https://ia801402.us.archive.org/11/items/MIT6.0001F16/MIT6_0001F16_Lecture_01_300k.mp4"
    # TEST_URL = (
    #     "https://archive.org/download/ClaytonCameron_2013Y/ClaytonCameron_2013Y.mp4"
    # )
    # with slay.run_local(secrets={"baseten_api_key": os.environ["B10_PERSONAL_KEY"]}):
    #     transcribe = Transcribe()
    #     result = asyncio.run(transcribe.run(TEST_URL))
    #     print(result)

    remote = slay.deploy_remotely(ChunkedTranscribe, workflow_name="Demo")
