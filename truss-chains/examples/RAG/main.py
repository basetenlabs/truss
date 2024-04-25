import logging
import os

import pydantic
import truss_chains as chains
from truss_chains import utils

vlog = logging.getLogger("vlog")
vlog.handlers = []
log_format = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
date_format = "%m%d %H:%M:%S"
formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
handler = logging.FileHandler("/tmp/rag-log.txt")
handler.setFormatter(formatter)
vlog.addHandler(handler)


_CORPUS_NAME = "test_collection"


class QueryParams(pydantic.BaseModel):
    num_context_docs: int


def _make_system_prompt(document_contents: list[str]) -> str:
    parts = [
        "You are tasked with helping software engineers, analysts or non-techincal "
        "users of then ML model hosting platform *Baseten*. You are given the "
        "following context-relevant resources to use for your response. "
        "Do not assume best practices or URLs that are not explicitly mentioned in"
        "these resources.\n\n"
    ]
    for i, content in enumerate(document_contents):
        parts.append(f"**RESOURCE {i}**:.\n\n{content}\n\n")
    parts.append(
        "Provide an actionable response with examples. Add references or links to "
        "documentation if possible."
        "If possible generate examples that are suitable for copy and paste and "
        "can be run directly, i.e. self-contained without need to add anything."
    )
    return "\n".join(parts)


class ClaudeClient:
    def __init__(self, api_key: str):
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)

    async def run(self, query: str, system_prompt: str) -> str:
        messages = [{"role": "user", "content": query}]
        vlog.info("Querying Claude.")
        message = self._client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            messages=messages,
            system=system_prompt,
        )
        return message.content[0].text


class OpenaiClient:
    def __init__(self, api_key: str):
        import openai

        self._client = openai.Client(api_key=api_key)

    async def run(self, query: str, system_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        vlog.info("Querying GPT4.")
        completion = self._client.chat.completions.create(
            model="gpt-4-turbo-2024-04-09",
            messages=messages,
        )
        return completion.choices[0].message.content


class Llama70B(chains.StubBase):
    async def run(self, query: str, system_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        vlog.info("Querying Llama.")
        json_payload = {
            "messages": messages,
            "stream": False,
            "max_new_tokens": 512,
            # "temperature": 0.9,
        }
        resp = await self._remote.predict_async(json_payload)
        return resp


class VectorStore(chains.ChainletBase):

    remote_config = chains.RemoteConfig(
        docker_image=chains.DockerImage(
            pip_requirements_file=chains.make_abs_path_here("requirements.txt"),
            pip_requirements=["chromadb"],
            data_dir=chains.make_abs_path_here("vector_data/"),
        )
    )

    def __init__(self, context: chains.DeploymentContext = chains.provide_context()):
        super().__init__(context)
        import chromadb

        path = os.path.join(context.data_dir, "chroma")
        settings = chromadb.Settings(is_persistent=True, anonymized_telemetry=False)
        self._chroma_client = chromadb.PersistentClient(path=path, settings=settings)
        self._chroma_collection = self._chroma_client.get_collection(_CORPUS_NAME)

    async def run(self, query: str, params: QueryParams) -> list[str]:
        query_result = self._chroma_collection.query(
            query_texts=[query],
            n_results=params.num_context_docs,
            include=["documents"],
        )
        multi_documents = query_result["documents"]
        assert multi_documents is not None
        documents = utils.expect_one(multi_documents)
        vlog.info(f"Retrieved Documents for Query\n{query}:")
        for i, d in enumerate(documents):
            vlog.info(f"{i}:\n{d}")
        return documents


# Llama 3 70B Instruct
Llama70B_URL = "https://model-4w57z7r3.api.baseten.co/production/predict"


class RAG(chains.ChainletBase):

    remote_config = chains.RemoteConfig(
        docker_image=chains.DockerImage(
            pip_requirements_file=chains.make_abs_path_here("requirements.txt"),
            pip_requirements=["anthropic"],
        ),
        assets=chains.Assets(secret_keys=["anthropic_api_key"]),
    )

    def __init__(
        self,
        context: chains.DeploymentContext = chains.provide_context(),
        vector_store: VectorStore = chains.provide(VectorStore),
    ):
        super().__init__(context)
        self._vector_store = vector_store
        # self._llm = Llama70B.from_url(Llama70B_URL, context)
        self._llm = ClaudeClient(context.secrets["anthropic_api_key"])
        # self._llm = OpenaiClient(context.secrets["openai_api_key"])

    async def run(self, query: str, params: QueryParams) -> str:
        # refining_query = (
        #     f"{query}.\n\nIn order to get help with this request, what kind "
        #     "of information would you need to research? At which documentation "
        #     "pages and for which keywords would you search?"
        # )
        # enriched_query = await self._llm.run(refining_query, "")
        # vlog.info(enriched_query)
        document_contents = await self._vector_store.run(query, params)

        system_prompt = _make_system_prompt(document_contents)
        vlog.info(f"System Prompt:\n{system_prompt}")
        vlog.info(f"Query: {query}")
        answer = await self._llm.run(query, system_prompt)
        return answer


if __name__ == "__main__":
    import asyncio

    from rich.console import Console

    console = Console()

    with chains.run_local(
        secrets={
            "baseten_chain_api_key": os.environ["BASETEN_API_KEY"],
            "anthropic_api_key": os.environ["ANTHROPIC_API_KEY"],
            "openai_api_key": os.environ["OPENAI_API_KEY"],
        },
        data_dir=chains.make_abs_path_here("vector_data").abs_path,
    ):
        rag = RAG()
        # test_query = "How can I programmatically change auto-scaling settings?"
        # test_query = "What can you tell me about the `__init__` method?"
        test_query = "How do I improve my coldstarts?"
        # test_query = "What's the coolest feature of baseten?"
        test_params = QueryParams(num_context_docs=5)
        result = asyncio.get_event_loop().run_until_complete(
            rag.run(test_query, test_params)
        )
        console.print(result)
