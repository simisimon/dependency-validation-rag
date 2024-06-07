from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.postprocessor import LLMRerank, SentenceTransformerRerank
from llama_index.postprocessor.rankgpt_rerank import RankGPTRerank
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import NodeWithScore
from llama_index_client import MetadataFilters
from typing import List, Any
  

class CustomRerankRetriever(BaseRetriever):
    def __init__(
        self, 
        vector_store: PineconeVectorStore, 
        embed_model: Any,
        similarity_top_k: int
    ) -> None:
        super().__init__()
        self._vector_retriever = VectorIndexRetriever(
            index=VectorStoreIndex.from_vector_store(vector_store=vector_store),
            embed_model=embed_model,
            similarity_top_k=similarity_top_k
        )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:

        retrieved_nodes = self._vector_retriever.retrieve(query_bundle)
        print("Len of retrieved nodes: ", len(retrieved_nodes))

        # LLMReranker
        #reranker = LLMRerank(service_context=self._service_context)
        #retrieved_nodes = reranker.postprocess_nodes(retrieved_nodes, query_bundle)

        # SentenceTransformerRerank
        reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=3)
        retrieved_nodes = reranker.postprocess_nodes(nodes=retrieved_nodes, query_bundle=query_bundle)
        print("Len of retrieved nodes after reranking: ", len(retrieved_nodes))

        #reranker = RankGPTRerank(llm=Settings.llm, top_n=3, verbose=True)
        #retrieved_nodes = reranker.postprocess_nodes(nodes=retrieved_nodes, query_bundle=query_bundle)
        #print("Len of retrieved nodes after reranking: ", len(retrieved_nodes))

        return retrieved_nodes



class CustomRerankAndFilterRetriever(BaseRetriever):
    def __init__(
        self, 
        vector_store: PineconeVectorStore, 
        embed_model: Any,
        similarity_top_k: int,
        filters: MetadataFilters
    ) -> None:
        super().__init__()
        self._vector_retriever = VectorIndexRetriever(
            index=VectorStoreIndex.from_vector_store(vector_store=vector_store),
            embed_model=embed_model,
            similarity_top_k=similarity_top_k,
            filters=filters
        )
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        retrieved_nodes = self._vector_retriever.retrieve(query_bundle)
        print("Len of retrieved nodes: ", len(retrieved_nodes))
        return retrieved_nodes