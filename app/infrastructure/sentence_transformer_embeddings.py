
from typing import Sequence

from sentence_transformers import SentenceTransformer

from app.domain.embeddings import Vector


class SentenceTransformerEmbeddingProvider:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 8,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.model_name = model_name
        self.batch_size = batch_size

        self._model = SentenceTransformer(
            model_name,
            device=device,
        )

    def encode(
        self,
        texts: Sequence[str],
    ) -> list[Vector]:
        if not texts:
            return []

        embeddings = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return [
            tuple(float(value) for value in vector)
            for vector in embeddings
        ]
