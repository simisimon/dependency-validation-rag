from llama_index.core.postprocessor import SentenceTransformerRerank, LLMRerank
from llama_index.core.schema import NodeWithScore
from llama_index.core import Settings, PromptTemplate, VectorStoreIndex, StorageContext
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from weaviate.classes.config import Configure, VectorDistances
from weaviate import WeaviateClient
from typing import List
from rich.logging import RichHandler
import weaviate
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()],
)


REWRITE_QUERY = PromptTemplate(
    "You are a helpful assistant that generates multiple search queries \
    to provide information about the configuration options mentioned in the input query, \
    such as descriptions and prior usages of the configuration options.\n"
    "Generate {num_queries} search queries, one on each line, \
    for both configuration options mentioned in the following input query: {query}"
)


class RetrievalEngine:
    def __init__(
        self, 
        weaviate_client: str,
        search_strategy: str,
        distance_metrics: str,
        top_k: int,
        top_n: int,
        num_queries: int,
        alpha: float
    ) -> None:
        self._weaviate_client = weaviate_client
        self.search_strategy = search_strategy
        self.distance_metric = distance_metrics
        self.top_k = top_k
        self.top_n = top_n
        self.num_queries = num_queries
        self.alpha = alpha

        self._llm = Settings.llm
        self._embed_model = Settings.embed_model    
    
    def _get_distance_metric(self, metric) -> VectorDistances:
        """
        Get distance metric
        """
        distance_metric = None
        if metric == "dot":
            return VectorDistances.DOT
        if metric == "cosine":
            return  VectorDistances.COSINE
        if metric == "hamming":
            return VectorDistances.HAMMING
        if metric == "mnahatten":
            return VectorDistances.MANHATTAN
        if metric == "squared":
            return VectorDistances.L2_SQUARED
        
        return distance_metric


    def _create_reranker(self, rerank: str):
        """
        Create reranker.
        """
        reranker = None
        if rerank == "llm":
            reranker =  LLMRerank(
                choice_batch_size=self.top_n,
                top_n=self.top_n)
        if rerank == "sentence":
            reranker = SentenceTransformerRerank(
                model="cross-encoder/ms-marco-MiniLM-L-2-v2",
                top_n=self.top_n
            )

        return reranker


    def _rewrite_queries(self, query_str: str, num_queries: str) -> List[str]:
        """
        Rewrite query. Return list of rewritten queries. 
        """
        response = Settings.llm.predict(
            prompt=REWRITE_QUERY, 
            num_queries=num_queries,
            query=query_str
        )

        queries = response.split("\n")
        return queries


    def retrieve(
        self, 
        index_name: str,
        query_str: str,
        rerank: str
    ) -> List[NodeWithScore]:
        """
        Retrieve context.
        """
        # rewrite queries
        queries = self._rewrite_queries(
            query_str=query_str,
            num_queries=self.num_queries
        )

        # create reranker
        reranker = self._create_reranker(rerank=rerank)

        # create weaviate vector store
        vector_store = WeaviateVectorStore(
            weaviate_client=self._weaviate_client, 
            index_name=index_name
        )

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )


        # create vector store index
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=self._get_distance_metric(self.distance_metric)
            )
        )

        self._weaviate_client.close()

        # create RetrieverQueryEngine
        query_engine = index.as_query_engine(
            vector_store_query_mode=self.search_strategy,
            similarity_top_k=self.top_k,
            alpha=self.alpha,
        )

        with self._weaviate_client:
            return query_engine.query(query_str) 

        #return query_engine.retrieve(query_bundle=QueryBundle(query_str=query_str))


        # retrieve nodes based on rewritten query
        #all_retrieved_nodes = []
        #for index, query in enumerate(queries):
        #    retrieved_nodes = query_engine.retrieve(query_bundle=QueryBundle(query_str=query))
        #    logging.info(f"Len of retrieved nodes of query {index}: {len(retrieved_nodes)}")
        #    all_retrieved_nodes += retrieved_nodes

        # rerank retrieved nodes
        #reranked_nodes = reranker.postprocess_nodes(
        #    nodes=all_retrieved_nodes,
        #    query_bundle=query_str
        #)
        #logging.info(f"Len of retrieved nodes after reranking: {len(reranked_nodes)}")


        #return reranked_nodes
    

