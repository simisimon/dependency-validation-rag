from llama_index.core.postprocessor import SentenceTransformerRerank, LLMRerank
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core import Settings, VectorStoreIndex
from pinecone import Pinecone
from typing import List
from prompt_templates import REWRITE_QUERY, RELEVANCE_PROMPT
from rich.logging import RichHandler
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()],
)


class RetrievalEngine:
    def __init__(
        self, 
        pinecone_client: Pinecone,
        with_rewriting: bool,
        rerank: str, 
        top_k: int,
        top_n: int,
        alpha: float,
    ) -> None:
        logging.info(f"Retrieval engine initialized.")
        self._pinecone_client = pinecone_client
        self.with_rewriting = with_rewriting
        self.rerank = rerank
        self.top_k = top_k
        self.top_n = top_n
        self.alpha = alpha  


    def _get_vector_store(self, index_name: str) -> PineconeVectorStore:
        """
        Get Pinecone vector store.
        """
        if index_name not in self._pinecone_client.list_indexes().names():
            logging.error(f"Index {index_name}  does not exist.")
            raise Exception()
        
        logging.info(f"Select index: {index_name}.")
        index = self._pinecone_client.Index(index_name)
        vector_store = PineconeVectorStore(
            pinecone_index=index,
            add_sparse_vector=True
        )

        return vector_store


    def _create_reranker(self):
        """
        Create reranker.
        """
        reranker = None

        if self.rerank == "llm":
            reranker =  LLMRerank(
                llm=Settings.llm,
                choice_batch_size=self.top_n,
                top_n=self.top_n)
        if self.rerank == "sentence":
            reranker = SentenceTransformerRerank(
                model="cross-encoder/ms-marco-MiniLM-L-2-v2",
                top_n=self.top_n
            )

        return reranker
    

    def _rewrite_queries(self, query_str: str) -> List[str]:
        """
        Rewrite query. Return list of rewritten queries. 
        """
        logging.info(f"Rewrite query in one query.")
        response = Settings.llm.predict(
            prompt=REWRITE_QUERY,
            query=query_str
        )

        queries = response.split("\n")

        print("Queries: ", queries, len(queries))
        return queries
    

    def _relevance_filter(self, query_str: str, node: NodeWithScore) -> bool:
        """
        Check if the retrieved context is relevant to the query.
        """
        response = Settings.llm.predict(
            prompt=RELEVANCE_PROMPT,
            query_str=query_str,
            context_str=node.text
        )

        if "yes" in response.lower():
            passing = True
        else:
            passing = False

        return passing
    

    def _rerank_nodes(self, nodes: List[NodeWithScore], query_str: str) -> List[NodeWithScore]:
        """
        Rerank retrieved nodes.
        """
        reranker = self._create_reranker()
        duplicates = True
        
        while duplicates:
            reranked_nodes = reranker.postprocess_nodes(
                nodes=nodes,
                query_bundle=QueryBundle(query_str=query_str)
            )

            node_ids = set(node.node_id for node in reranked_nodes)
            print(node_ids)
            if len(node_ids) == 3:
                print("Rerank done.")
                duplicates = False
            else:
                print("Reranking again.")
                duplicates = True

        logging.info(f"Rerank {len(nodes)} retrieved nodes into {len(reranked_nodes)} nodes.")

        return reranked_nodes


    def retrieve(self, index_name: str, query_str: str) -> List[NodeWithScore]:
        """
        Retrieve context.
        """
        if index_name == "all":
            retrieved_nodes = []
            for name in self._pinecone_client.list_indexes().names():
                vector_store = self._get_vector_store(index_name=name)
                nodes = self._retrieve(vector_store=vector_store, query_str=query_str)
                retrieved_nodes += nodes

            reranked_retrieved_nodes = self._rerank_nodes(nodes=retrieved_nodes, query_str=query_str)
            
            return reranked_retrieved_nodes

        else:
            vector_store = self._get_vector_store(index_name=index_name)
            retrieved_nodes = self._retrieve(vector_store=vector_store, query_str=query_str)
            reranked_retrieved_nodes = self._rerank_nodes(nodes=retrieved_nodes, query_str=query_str)
            return reranked_retrieved_nodes

    def _retrieve(self, vector_store: PineconeVectorStore, query_str: str) -> List[NodeWithScore]:
        """
        Retrieve context from vector store.
        """
        # create vector store index
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=Settings.embed_model,
        )

        # create RetrieverQueryEngine
        query_engine = index.as_query_engine(
            vector_store_query_mode="hybrid",
            similarity_top_k=self.top_k,
            alpha=self.alpha,
        )

        if self.with_rewriting:
            logging.info("Retrieve relevant context with query rewriting.")
            queries = self._rewrite_queries(query_str=query_str)
       
            retrieved_nodes = []
            for index, query in enumerate(queries):
                nodes = query_engine.retrieve(
                    query_bundle=QueryBundle(query_str=query)
                )
                retrieved_nodes += nodes

            return retrieved_nodes
        
        logging.info("Retrieve relevant context without query rewriting.")
        retrieved_nodes = query_engine.retrieve(
            query_bundle=QueryBundle(query_str=query_str)
        )

        return retrieved_nodes
        
