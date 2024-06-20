
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever, AutoMergingRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index_client import MetadataFilters
from llama_index.core.schema import NodeWithScore
from llama_index_client import MetadataFilters, MetadataFilter
from llama_index.core.indices.query.schema import QueryBundle
from llama_index.core import VectorStoreIndex, Settings, StorageContext

from typing import List, Any
from rich.logging import RichHandler
from prompt_templates import QUERY_GEN_PROMPT, RELEVANCE_PROMPT
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()],
)


class RetrieverFactory:
    @staticmethod
    def get_retriever(retriever_type: str, vector_store, top_k: int) -> List[str]:
        """
        Get Retriever.
        """
        if retriever_type == "rerank":
            return CustomRerankRetriever(
                vector_store=vector_store,
                embed_model=Settings.embed_model,
                similarity_top_k=top_k
            )

        if retriever_type == "rerank_and_filter":
            filters = [
                MetadataFilter(
                    key='technology',
                    value=title,
                    operator='==',
                
                )
                for title in ['docker', 'spring-boot']
            ]

            filters = MetadataFilters(filters=filters, condition="or")

            return CustomRerankAndFilterRetriever(
                vector_store=vector_store,
                embed_model=Settings.embed_model,
                similarity_top_k=top_k,
                filters=filters
            )

        if retriever_type == "auto_merging":
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            base_retriever = VectorIndexRetriever(
                index=VectorStoreIndex.from_vector_store(vector_store=vector_store),
                similarity_top_k=top_k,
                embed_model=Settings.embed_model
            )
            return AutoMergingRetriever(vector_retriever=base_retriever, storage_context=storage_context)

        if retriever_type == "rerank_and_relevance_retriever":
            return CustomRerankAndRelevanceRetriever(
                vector_store=vector_store,
                embed_model=Settings.embed_model,
                similarity_top_k=top_k
            )

        if retriever_type == "hybrid":
            return CustomHybridRetriever(
                vector_store=vector_store,
                embed_model=Settings.embed_model,
                similarity_top_k=top_k
            )
        
        return None


class CustomBaseRetriever(BaseRetriever):
    def __init__(
        self, 
        vector_store: PineconeVectorStore, 
        embed_model: Any,
        similarity_top_k: int
    ):
        super().__init__()
        self._vector_retriever = VectorIndexRetriever(
            index=VectorStoreIndex.from_vector_store(vector_store=vector_store),
            embed_model=embed_model,
            similarity_top_k=similarity_top_k
        )
        


    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        all_retrieved_nodes = []
        queries = self._generate_queries(
            query_bundle=query_bundle,
            num_queries=3
        )

        for index, query in enumerate(queries):
            retrieved_nodes = self._vector_retriever.retrieve(QueryBundle(query_str=query))
            logging.info(f"Len of retrieved nodes of query {index}: {len(retrieved_nodes)}")
            all_retrieved_nodes += retrieved_nodes[:1]

        return all_retrieved_nodes
    
    def _rerank_nodes(self, nodes: List[NodeWithScore], query_bundle: QueryBundle, top_n: int = 3) -> List[NodeWithScore]:
        reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=top_n)
        reranked_nodes = reranker.postprocess_nodes(nodes=nodes, query_bundle=query_bundle)
        logging.info(f"Len of retrieved nodes after reranking: {len(reranked_nodes)}")
        return reranked_nodes


    def _generate_queries(self, query_bundle: QueryBundle, num_queries: int) -> List[str]:
        """
        Rewrite query. Return list of rewritten queries. 
        """
        response = Settings.llm.predict(
            prompt=QUERY_GEN_PROMPT, 
            num_queries=num_queries,
            query=query_bundle.query_str
        )

        queries = response.split("\n")
        return queries


    def _relevance_filter(self, query_bundle: QueryBundle, node: NodeWithScore) -> bool:
        """
        Check if the query is in line with the retrieved context.
        """
        response = Settings.llm.predict(
            prompt=RELEVANCE_PROMPT,
            query_str=query_bundle.query_str,
            context_str=node.text
        )

        if "yes" in response.lower():
            passing = True
        else:
            passing = False

        return passing


class CustomRerankRetriever(CustomBaseRetriever):
    def __init__(
        self, 
        vector_store: PineconeVectorStore, 
        embed_model: Any,
        similarity_top_k: int
    ) -> None:
        logging.info("Custom Rerank Retriever initialized.")
        super().__init__(vector_store, embed_model, similarity_top_k)
        self._vector_retriever = VectorIndexRetriever(
            index=VectorStoreIndex.from_vector_store(vector_store=vector_store),
            embed_model=embed_model,
            similarity_top_k=similarity_top_k
        )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        all_retrieved_nodes = []
        queries = self._generate_queries(
            query_bundle=query_bundle,
            num_queries=3
        )

        for index, query in enumerate(queries):
            retrieved_nodes = self._vector_retriever.retrieve(QueryBundle(query_str=query))
            logging.info(f"Len of retrieved nodes of query {index}: {len(retrieved_nodes)}")
            all_retrieved_nodes += retrieved_nodes


        reranked_nodes = self._rerank_nodes(
            nodes=all_retrieved_nodes,
            query_bundle=query_bundle
        )
    
        return reranked_nodes
    

class CustomRerankAndRelevanceRetriever(CustomBaseRetriever):
    def __init__(
        self, 
        vector_store: PineconeVectorStore, 
        embed_model: Any,
        similarity_top_k: int
    ) -> None:
        logging.info("Custom Rerank and Relevance Retriever initialized.")
        super().__init__(vector_store, embed_model, similarity_top_k)
        self._vector_retriever = VectorIndexRetriever(
            index=VectorStoreIndex.from_vector_store(vector_store=vector_store),
            embed_model=embed_model,
            similarity_top_k=similarity_top_k
        )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        all_retrieved_nodes = []
        queries = self._generate_queries(
            query_bundle=query_bundle,
            num_queries=3
        )

        for index, query in enumerate(queries):
            retrieved_nodes = self._vector_retriever.retrieve(QueryBundle(query_str=query))
            logging.info(f"Len of retrieved nodes of query {index}: {len(retrieved_nodes)}")
            filtered_nodes = [node for node in retrieved_nodes if self._relevance_filter(QueryBundle(query_str=query), node)]
            logging.info(f"Len of filtered nodes of query {index}: {len(filtered_nodes)}")
            all_retrieved_nodes += filtered_nodes
            
        reranked_nodes = self._rerank_nodes(
            nodes=all_retrieved_nodes,
            query_bundle=query_bundle
        )
    
        return reranked_nodes
    


class CustomRerankAndFilterRetriever(CustomBaseRetriever):
    def __init__(
        self, 
        vector_store: PineconeVectorStore, 
        embed_model: Any,
        similarity_top_k: int,
        filters: MetadataFilters
    ) -> None:
        logging.info("Custom Rerank and Filter Retriever initialized.")
        super().__init__(vector_store, embed_model, similarity_top_k)
        self._vector_retriever = VectorIndexRetriever(
            index=VectorStoreIndex.from_vector_store(vector_store=vector_store),
            embed_model=embed_model,
            similarity_top_k=similarity_top_k,
            filters=filters
        )
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        all_retrieved_nodes = []
        queries = self.generate_queries(
            query_bundle=query_bundle,
            num_queries=3
        )

        for index, query in enumerate(queries):
            retrieved_nodes = self._vector_retriever.retrieve(QueryBundle(query_str=query))
            logging.info(f"Len of retrieved nodes of query {index}: {len(retrieved_nodes)}")
            all_retrieved_nodes += retrieved_nodes

        reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=3)
        final_retrieved_nodes = reranker.postprocess_nodes(nodes=all_retrieved_nodes, query_bundle=query_bundle)
        logging.info(f"Len of retrieved nodes after reranking: {len(retrieved_nodes)}")

        return final_retrieved_nodes


class CustomHybridRetriever(CustomBaseRetriever):
    def __init__(
        self, 
        vector_store: PineconeVectorStore, 
        embed_model: Any,
        similarity_top_k: int
    ):
        logging.info("Custom Hybrid Retriever initialized.")
        super().__init__(vector_store, embed_model, similarity_top_k)
        self._vector_store_index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
            similarity_top_k=similarity_top_k
        )
        self._top_k = similarity_top_k

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_engine = self._vector_store_index.as_query_engine(
            vector_store_query_mode="hybrid",
            similarity_top_k=self._top_k
        )

        all_retrieved_nodes = []
        queries = self._generate_queries(
            query_bundle=query_bundle,
            num_queries=3
        )

        for index, query in enumerate(queries):
            retrieved_nodes = query_engine.retrieve(query_bundle=QueryBundle(query_str=query))
            logging.info(f"Len of retrieved nodes of query {index}: {len(retrieved_nodes)}")
            all_retrieved_nodes += retrieved_nodes

        reranked_nodes = self._rerank_nodes(
            nodes=all_retrieved_nodes,
            query_bundle=query_bundle
        )

        return reranked_nodes
