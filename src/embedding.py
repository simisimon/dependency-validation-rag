from typing import Any, List
from sentence_transformers import SentenceTransformer

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding


class SentenceTransformerEmbedding(BaseEmbedding):
    _model: SentenceTransformer = PrivateAttr()

    def __init__(
        self,
        model_name: str,
        **kwargs: Any,
    ) -> None:
        self._model = SentenceTransformer(model_name)
        super().__init__(**kwargs)

    @classmethod
    def class_name(cls) -> str:
        return cls.model_name

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        embeddings = self._model.encode(query)
        return embeddings

    def _get_text_embedding(self, text: str) -> List[float]:
        embeddings = self._model.encode(text)
        return embeddings

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(
            [[text] for text in texts]
        )
        return embeddings