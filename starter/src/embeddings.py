"""Embedding adapters with an explicit, reproducible model selection."""
from __future__ import annotations

import os
from typing import Protocol


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """Use OpenAI embeddings or a named local Sentence Transformers model.

    No silent fallback is allowed: all documents and queries in a vector index
    must use the same real embedding model.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.client = None
        self.model = None
        if self.model_name.startswith("text-embedding-"):
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("OpenAI embeddings require `python -m pip install openai`") from exc
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is required for the selected embedding model")
            self.client = OpenAI(api_key=api_key)
            # OpenAI text-embedding-3-small defaults to 1536 dimensions.
            self.dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
        else:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("Local embeddings require `python -m pip install sentence-transformers`") from exc
            self.model = SentenceTransformer(self.model_name)
            self.dimensions = int(self.model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.client is not None:
            response = self.client.embeddings.create(model=self.model_name, input=texts, dimensions=self.dimensions)
            return [list(map(float, row.embedding)) for row in response.data]
        if self.model is not None:
            embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return [list(map(float, vector)) for vector in embeddings]
        raise RuntimeError("No embedding provider initialized")

    def embed_query(self, question: str) -> list[float]:
        if self.client is not None:
            response = self.client.embeddings.create(model=self.model_name, input=[question], dimensions=self.dimensions)
            return list(map(float, response.data[0].embedding))
        if self.model is not None:
            query_vector = self.model.encode([question], show_progress_bar=False)
            return [float(value) for value in query_vector[0]]
        raise RuntimeError("No embedding provider initialized")
