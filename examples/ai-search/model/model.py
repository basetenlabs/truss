from typing import Dict, List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._engine = None
        self._db = None

    def load(self):
        # Load model here and assign to self._model.
        self._engine = SentenceTransformer("all-MiniLM-L6-v2")
        self._db = QdrantClient(":memory:")
        self._db.recreate_collection(
            collection_name="my_collection",
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

        # Our sentences we like to encode
        sentences = [
            "This framework generates embeddings for each input sentence",
            "Sentences are passed as a list of string.",
            "The quick brown fox jumps over the lazy dog.",
        ]
        embeddings = self._engine.encode(sentences)
        self._db.upsert(
            collection_name="my_collection",
            points=[
                PointStruct(id=idx, vector=vector.tolist(), payload={"text": s})
                for idx, (vector, s) in enumerate(zip(embeddings, sentences))
            ],
        )

    def predict(self, model_input: Dict) -> List:
        query_text = model_input.pop("query")
        limit = model_input.get("limit", 1)

        query_vector = self._engine.encode(query_text)
        hits = self._db.search(
            collection_name="my_collection", query_vector=query_vector, limit=limit
        )
        results = []
        for hit in hits:
            results.append(
                {
                    "payload": hit.payload,
                    "score": hit.score,
                }
            )
        return results
