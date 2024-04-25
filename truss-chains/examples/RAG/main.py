import anthropic
import chromadb
import pydantic
import truss_chains as chains
from coverage.annotate import os
from truss_chains import utils

IMAGE_VECTORSTORE = chains.DockerImage(
    pip_requirements_file=chains.make_abs_path_here("requirements.txt"),
    pip_requirements=["chromadb"],
    data_dir=chains.make_abs_path_here("vector_data/"),
)


class QueryParams(pydantic.BaseModel):
    num_context_docs: int = 2


def _make_system_prompt(document_contents: list[str]) -> str:
    parts = [
        "You are tasked with helping software engineers, users and other people "
        "using a machine learning model hosting platform 'Baseten'.  Given the"
        "following information:\n\n"
    ]
    for i, content in enumerate(document_contents):
        parts.append(f"{i}.\n{content}\n\n")
    parts.append(
        "Give a useful actionable answer with examples to the "
        "users request. Add references or links to documentation if possible."
        "If possible generate examples that are suitable for copy and paste and can be run directly without need to add anything."
    )
    return "\n".join(parts)


class ClaudeClient:
    def __init__(self, api_key: str):
        self._client = anthropic.Anthropic(api_key=api_key)

    async def run(self, query: str, system_prompt: str) -> str:
        messages = [
            {"role": "user", "content": query},
        ]
        message = self._client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            messages=messages,
            system=system_prompt,
        )
        return message.content[0].text


class Llama70B(chains.StubBase):
    """Treat the whisper model like an external third party tool."""

    async def run(self, query: str, system_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        json_payload = {
            "messages": messages,
            "stream": False,
            "max_new_tokens": 512,
            # "temperature": 0.9,
        }
        # print(messages[0])
        resp = await self._remote.predict_async(json_payload)
        # TODO: strip special tokens?
        return resp


class VectorStore(chains.ChainletBase):

    remote_config = chains.RemoteConfig(docker_image=IMAGE_VECTORSTORE)

    def __init__(self, context: chains.DeploymentContext = chains.provide_context()):
        super().__init__(context)
        path = os.path.join(context.data_dir, "chroma")
        print(path)
        settings = chromadb.Settings()
        self._chroma_client = chromadb.PersistentClient(path=path)
        self._chroma_collection = self._chroma_client.get_collection("test_collection")

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
Llama70B_URL = "https://model-4w57z7r3.api.baseten.co/production/predict"


class RAG(chains.ChainletBase):

    remote_config = chains.RemoteConfig(
        docker_image=chains.DockerImage(
            pip_requirements_file=chains.make_abs_path_here("requirements.txt")
        )
    )

    def __init__(
        self,
        context: chains.DeploymentContext = chains.provide_context(),
        vector_stor: VectorStore = chains.provide(VectorStore),
    ):
        super().__init__(context)
        self._vector_store = vector_stor
        # self._llm = Llama70B.from_url(Llama70B_URL, context)
        self._llm = ClaudeClient(context.secrets["anthropic_api_key"])

    async def run(self, query: str, params: QueryParams) -> list[str]:
        # TODO: ask LLM what is needed to answer query.

        # refining_query = (
        #     f"{query}.\n\nIn order to get help with this request, what kind "
        #     "of information would you need to research? At which documentation "
        #     "pages and for which keywords would you search?"
        # )
        # enriched_query = await self._llm.run(refining_query, "")
        # print(enriched_query)
        document_contents = await self._vector_store.run(query, params)

        system_prompt = _make_system_prompt(document_contents)
        print(f"System Prompt:\n{system_prompt}")
        print(f"Query: {query}")
        answer = await self._llm.run(query, system_prompt)
        return answer


if __name__ == "__main__":
    import asyncio

    with chains.run_local(
        secrets={
            "baseten_chain_api_key": os.environ["BASETEN_API_KEY"],
            "anthropic_api_key": os.environ["ANTHROPIC_API_KEY"],
        },
        data_dir=chains.make_abs_path_here("vector_data").abs_path,
    ):
        rag = RAG()
        test_query = "How can I programmatically change auto-scaling settings?"
        params = QueryParams(num_context_docs=3)
        result = asyncio.get_event_loop().run_until_complete(
            rag.run(test_query, params)
        )
        print(result)
