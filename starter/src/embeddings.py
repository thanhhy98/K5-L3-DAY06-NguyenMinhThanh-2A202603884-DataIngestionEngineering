"""Choose a real embedding provider (API or local) and implement this contract."""
from __future__ import annotations

from typing import Protocol

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - fallback for minimal local setup
    SentenceTransformer = None


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
    """Use a local semantic model when available, otherwise a TF-IDF fallback."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        if SentenceTransformer is not None:
            self.model = SentenceTransformer(model_name)
            demo_vector = self.model.encode(["sample text"], show_progress_bar=False)
            self.dimensions = len(demo_vector[0])
            self._vectorizer = None
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.model_name = "tfidf-fallback"
            self.model = None
            self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            self._vectorizer.fit(["sample text"])
            demo_vector = self._vectorizer.transform(["sample text"]).toarray()[0]
            self.dimensions = len(demo_vector)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.model is not None:
            embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return [list(map(float, vector)) for vector in embeddings]
        matrix = self._vectorizer.transform(texts)
        return [list(map(float, row)) for row in matrix.toarray()]

    def embed_query(self, question: str) -> list[float]:
        if self.model is not None:
            query_vector = self.model.encode([question], show_progress_bar=False)
            return [float(value) for value in query_vector[0]]
        vector = self._vectorizer.transform([question]).toarray()[0]
        return [float(value) for value in vector]
