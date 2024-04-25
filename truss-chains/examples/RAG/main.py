import chromadb
import pydantic
import truss_chains as chains
from coverage.annotate import os
from truss_chains import utils

IMAGE_COMMON = chains.DockerImage(
    pip_requirements_file=chains.make_abs_path_here("requirements.txt"),
    pip_requirements=["chromadb"],
)


_SYSTEM_PROMPT = (
    "You are tasked with supporting engineers doing their work. "
    "You will be given some context information form documentation "
    "or chat history and have to give helpful answers to questions."
)


class QueryParams(pydantic.BaseModel):
    num_context_docs: int = 2


class LLMShim(chains.StubBase):
    """Treat the whisper model like an external third party tool."""

    async def run(self, query: str) -> str:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        json_payload = {
            "messages": messages,
            "stream": False,
            "max_new_tokens": 512,
            # "temperature": 0.9,
        }

        resp = await self._remote.predict_async(json_payload)
        return resp


class VectorStore(chains.ChainletBase):

    remote_config = chains.RemoteConfig(docker_image=IMAGE_COMMON)

    def __init__(self, context: chains.DeploymentContext = chains.provide_context()):
        super().__init__(context)
        path = os.path.join(context.data_dir, "corpus")
        self._chroma_client = chromadb.PersistentClient(path=path)
        self._chroma_collection = self._chroma_client.get_collection("default")

    async def run(self, query: str, params: QueryParams) -> list[str]:
        query_result = self._chroma_collection.query(
            query_texts=[query],
            n_results=params.num_context_docs,
            include=["documents"],
        )
        n_documents = query_result["documents"]
        assert n_documents is not None
        documents = utils.expect_one(n_documents)
        return documents


# Llama 3 70B Instruct
_LLM_URL = "https://model-4w57z7r3.api.baseten.co/production/predict"


class RAG(chains.ChainletBase):

    remote_config = chains.RemoteConfig(docker_image=IMAGE_COMMON)

    def __init__(
        self,
        context: chains.DeploymentContext = chains.provide_context(),
        vector_stor: VectorStore = chains.provide(VectorStore),
    ):
        super().__init__(context)
        self._vector_store = vector_stor
        self._llm = LLMShim.from_url(_LLM_URL, context)

    def _contextualize_query(
        self, query: str, document_contents: list[str], params: QueryParams
    ) -> str:
        parts = ["Given the following information:\n\n"]
        for i, content in enumerate(document_contents):
            parts.append(f"{i}.\n{content}\n\n")
        parts.append(
            f"Give a useful actionable answer with examples to the "
            f"following message: {query}"
        )
        return "\n".join(parts)

    async def run(self, query: str, params: QueryParams) -> list[str]:
        # TODO: ask LLM what is needed to answer query.
        document_contents = await self._vector_store.run(query, params)
        new_query = self._contextualize_query(query, document_contents, params)
        answer = await self._llm.run(new_query)
        return answer


if __name__ == "__main__":
    import asyncio

    with chains.run_local(
        secrets={"baseten_chain_api_key": os.environ["BASETEN_API_KEY"]},
        data_dir="/tmp",
    ):
        rag = RAG()
        query = "What's up?"
        params = QueryParams(num_context_docs=3)
        result = asyncio.get_event_loop().run_until_complete(rag.run(query, params))
        print(result)
