"""LangChain embeddings backed by fastembed (ONNX).

This implementation deliberately avoids scikit-learn and transformers, both of
which fail to import on machines where Windows Smart App Control blocks
scikit-learn's compiled extensions. fastembed runs the model through
onnxruntime instead, so no blocked native module is ever loaded.
"""

from __future__ import annotations

import numpy as np
from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings

DEFAULT_MODEL = "intfloat/multilingual-e5-large"


class FastEmbedEmbeddings(Embeddings):
    """LangChain ``Embeddings`` implementation backed by ``fastembed``.

    e5-family models expect distinct ``query:``/``passage:`` prefixes and
    L2-normalized vectors. ``HuggingFaceEmbeddings`` applied both implicitly,
    so they are applied explicitly here.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        cache_dir: str | None = None,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.batch_size = batch_size
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-12, None)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        passages = [self.passage_prefix + text for text in texts]
        vectors = np.asarray(
            list(self._model.embed(passages, batch_size=self.batch_size)),
            dtype=np.float32,
        )
        return self._normalize(vectors).tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = np.asarray(
            list(self._model.query_embed(self.query_prefix + text)),
            dtype=np.float32,
        )
        return self._normalize(vector)[0].tolist()
