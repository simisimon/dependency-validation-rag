
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.postprocessor import LLMRerank, SentenceTransformerRerank
from llama_index.postprocessor.rankgpt_rerank import RankGPTRerank
from llama_index_client import MetadataFilters
from llama_index.core.schema import NodeWithScore
from llama_index_client import MetadataFilters
from llama_index.core.indices.query.schema import QueryBundle
from llama_index.core import VectorStoreIndex, Settings
from typing import List, Any
from data import Dependency
from rich.logging import RichHandler
from prompt_templates import QUERY_GEN_PROMPT
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()],
)


def generate_queries(query_bundle: QueryBundle, num_queries: int) -> List[str]:
        response = Settings.llm.predict(
            prompt=QUERY_GEN_PROMPT, 
            num_queries=num_queries,
            query=query_bundle.query_str
        )

        queries = response.split("\n")
        return queries


class CustomBaseRetriever(BaseRetriever):
    def __init__(
        self, 
        vector_store: PineconeVectorStore, 
        embed_model: Any,
        similarity_top_k: int
    ):
        logging.info("Custom Base Retriever initialized.")
        super().__init__()
        self._vector_retriever = VectorIndexRetriever(
            index=VectorStoreIndex.from_vector_store(vector_store=vector_store),
            embed_model=embed_model,
            similarity_top_k=similarity_top_k
        )


    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        all_retrieved_nodes = []
        queries = generate_queries(
            query_bundle=query_bundle,
            num_queries=3
        )

        for query, index in enumerate(queries):
            retrieved_nodes = self._vector_retriever.retrieve(QueryBundle(query_str=query))
            logging.info(f"Len of retrieved nodes of query {index}: {len(retrieved_nodes)}")
            all_retrieved_nodes += retrieved_nodes[:1]

        return all_retrieved_nodes


class CustomRerankRetriever(BaseRetriever):
    def __init__(
        self, 
        vector_store: PineconeVectorStore, 
        embed_model: Any,
        similarity_top_k: int
    ) -> None:
        logging.info("Custom Rerank Retriever initialized.")
        super().__init__()
        self._vector_retriever = VectorIndexRetriever(
            index=VectorStoreIndex.from_vector_store(vector_store=vector_store),
            embed_model=embed_model,
            similarity_top_k=similarity_top_k
        )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        all_retrieved_nodes = []
        queries = generate_queries(
            query_bundle=query_bundle,
            num_queries=3
        )

        for query, index in enumerate(queries):
            retrieved_nodes = self._vector_retriever.retrieve(QueryBundle(query_str=query))
            logging.info(f"Len of retrieved nodes of query {index}: {len(retrieved_nodes)}")
            all_retrieved_nodes += retrieved_nodes


        # LLMReranker
        #reranker = LLMRerank(service_context=self._service_context)
          #final_retrieved_nodes = reranker.postprocess_nodes(nodes=all_retrieved_nodes, query_bundle=query_bundle)
        #logging.info(f"Len of retrieved nodes after reranking: {len(final_retrieved_nodes)}")


        # SentenceTransformerRerank
        reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=3)
        final_retrieved_nodes = reranker.postprocess_nodes(nodes=all_retrieved_nodes, query_bundle=query_bundle)
        logging.info(f"Len of retrieved nodes after reranking: {len(retrieved_nodes)}")

        #reranker = RankGPTRerank(llm=Settings.llm, top_n=3, verbose=True)
        #final_retrieved_nodes = reranker.postprocess_nodes(nodes=all_retrieved_nodes, query_bundle=query_bundle)
        #logging.info(f"Len of retrieved nodes after reranking: {len(final_retrieved_nodes)}")

        return final_retrieved_nodes


class CustomRerankAndFilterRetriever(BaseRetriever):
    def __init__(
        self, 
        vector_store: PineconeVectorStore, 
        embed_model: Any,
        similarity_top_k: int,
        filters: MetadataFilters
    ) -> None:
        logging.info("Custom Rerank and Filter Retriever initialized.")
        super().__init__()
        self._vector_retriever = VectorIndexRetriever(
            index=VectorStoreIndex.from_vector_store(vector_store=vector_store),
            embed_model=embed_model,
            similarity_top_k=similarity_top_k,
            filters=filters
        )
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        all_retrieved_nodes = []
        queries = generate_queries(
            query_bundle=query_bundle,
            num_queries=3
        )

        for query, index in enumerate(queries):
            retrieved_nodes = self._vector_retriever.retrieve(QueryBundle(query_str=query))
            logging.info(f"Len of retrieved nodes of query {index}: {len(retrieved_nodes)}")
            all_retrieved_nodes += retrieved_nodes

        reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=3)
        final_retrieved_nodes = reranker.postprocess_nodes(nodes=all_retrieved_nodes, query_bundle=query_bundle)
        logging.info(f"Len of retrieved nodes after reranking: {len(retrieved_nodes)}")

        return final_retrieved_nodes

