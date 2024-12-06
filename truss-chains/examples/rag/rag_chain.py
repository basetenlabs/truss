# This file uses Truss Chains to create and deploy an LLM with access to a vector database

import os

import truss_chains as chains

DOCUMENTS = [
    """
    Angela Martinez is a tech entrepreneur based in San Francisco.
    As the founder and CEO of a successful AI startup, she is a leading
    figure in the tech community. Outside of work, Angela enjoys hiking
    the trails around the Bay Area and volunteering at local animal shelters.
    """,
    """
    Ravi Patel resides in New York City, where he works as a financial analyst.
    Known for his keen insight into market trends, Ravi spends his weekends playing
    chess in Central Park and exploring the city's diverse culinary scene.
    """,
    """
    Sara Kim is a digital marketing specialist living in San Francisco.
    She helps brands build their online presence with creative strategies.
    Outside of work, Sara is passionate about photography and enjoys hiking
    the trails around the Bay Area.
    """,
    """
    David O'Connor calls New York City his home and works as a high school teacher.
    He is dedicated to inspiring the next generation through education. In his free time,
    David loves running along the Hudson River and participating in local theater productions.
    """,
    """
    Lena Rossi is an architect based in San Francisco. She designs sustainable and
    innovative buildings that contribute to the city's skyline. When she's not
    working, Lena enjoys practicing yoga and exploring art galleries.
    """,
    """
    Akio Tanaka lives in Tokyo and is a software developer specializing in mobile apps.
    Akio is an avid gamer and enjoys attending eSports tournaments.
    He also has a passion for cooking and often experiments with new recipes in his spare time.
    """,
    """
    Maria Silva is a nurse residing in New York City. She is dedicated to providing
    compassionate care to her patients. Maria finds joy in gardening and often
    spends her weekends tending to her vibrant flower beds and vegetable garden.
    """,
    """
    John Smith is a journalist based in San Francisco. He reports on international politics
    and has a knack for uncovering compelling stories. Outside of work, John is a history
    buff who enjoys visiting museums and historical sites.
    """,
    """
    Aisha Mohammed lives in Tokyo and works as a graphic designer. She creates visually stunning
    graphics for a variety of clients. Aisha loves to paint and often showcases her
    artwork in local exhibitions.
    """,
    """
    Carlos Mendes is an environmental engineer in San Francisco.
    He is passionate about developing sustainable solutions for urban areas.
    In his leisure time, Carlos enjoys surfing and participating in beach clean-up initiatives.
    """,
]


PERSON_MATCHING_PROMPT = (
    "You are matching alumni of a college to help them make connections. Explain why "
    "the person described first would want to meet the people selected from the "
    "matching database. Keep the reply brief to 1 to 3 sentences."
)

# Deploy the Phi-3 model from Baseten's model library:
# https://app.baseten.co/deploy/phi_3_mini_4k_instruct
# And insert the predict URL here (you can get it from the status page):
# E.g. "https://model-6wgeygoq.api.baseten.co/production/predict"
LLM_PREDICT_URL = ""


if not LLM_PREDICT_URL:
    raise ValueError("Please insert the predict URL for the Phi-3 model.")


# Create a Chainlet to serve as our vector database.
class VectorStore(chains.ChainletBase):
    # Add chromadb as a dependency for deployment.
    remote_config = chains.RemoteConfig(
        docker_image=chains.DockerImage(pip_requirements=["chromadb"])
    )

    # Runs once when the Chainlet is deployed or scaled up.
    def __init__(self):
        # Import Chainlet-specific dependencies in init, not at the top of
        # the file.
        import chromadb

        chroma_client = chromadb.EphemeralClient()
        self._collection = chroma_client.create_collection(name="bios")
        self._collection.add(
            documents=DOCUMENTS, ids=[f"id{n}" for n in range(len(DOCUMENTS))]
        )

    # Runs each time the Chainlet is called
    async def run_remote(self, query: str) -> list[str]:
        # This call to includes embedding the query string.
        results = self._collection.query(query_texts=[query], n_results=2)
        if results is None or not results:
            raise ValueError("No bios returned from the query")
        if not results["documents"] or not results["documents"][0]:
            raise ValueError("Bios are empty")
        return results["documents"][0]


class LLMClient(chains.StubBase):
    # Runs each time the Stub is called
    async def run_remote(self, new_bio: str, bios: list[str]) -> str:
        bios_info = "\n".join(bios)
        prompt = (
            f"{PERSON_MATCHING_PROMPT}\nPerson you're matching: {new_bio}\n"
            f"People from database: {bios_info}"
        )
        resp = await self.predict_async(
            {
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "max_new_tokens": 32,
            }
        )
        return resp["output"][len(prompt) :].strip()


@chains.mark_entrypoint
class RAG(chains.ChainletBase):
    def __init__(
        self,
        # Declare dependency chainlets.
        vector_store: VectorStore = chains.depends(VectorStore),
        context: chains.DeploymentContext = chains.depends_context(),
    ):
        self._vector_store = vector_store
        # The stub needs the context for setting up authentication.
        self._llm = LLMClient.from_url(LLM_PREDICT_URL, context)

    async def run_remote(self, new_bio: str) -> str:
        # Use the VectorStore Chainlet for context retrieval.
        bios = await self._vector_store.run_remote(new_bio)
        # Use the LLMClient Stub for augmented generation.
        contacts = await self._llm.run_remote(new_bio, bios)
        return contacts


if __name__ == "__main__":
    import asyncio

    with chains.run_local(
        # This secret is needed even locally, because part of this chain
        # calls the separately deployed Phi-3 model. Only the Chainlets
        # actually run locally.
        secrets={"baseten_chain_api_key": os.environ["BASETEN_API_KEY"]}
    ):
        rag_client = RAG()
        result = asyncio.get_event_loop().run_until_complete(
            rag_client.run_remote(
                """
                Sam just moved to Manhattan for his new job at a large bank.
                In college, he enjoyed building sets for student plays.
                """
            )
        )
        print(result)
