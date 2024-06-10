from pinecone import Pinecone, ServerlessSpec
from llama_index_client import MetadataFilter, MetadataFilters
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever, AutoMergingRetriever
from retriever import CustomBaseRetriever, CustomRerankAndFilterRetriever, CustomRerankRetriever
from typing import List
from rich.logging import RichHandler
import os
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()],
)


class RetrievalEngine:
    def __init__(self) -> None:
        logging.info("Retrieval Engine initialized.")

    def get_vector_store(
        self, 
        index_name: str, 
        dimension: int = 1536, 
        metric: str = "cosine"
    ):
        """
        Create and return vector store.
        """
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        if index_name not in pc.list_indexes().names():
            self.vector_db_instance.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

        index = pc.Index(index_name)
        vector_store = PineconeVectorStore(pinecone_index=index)

        return vector_store
    
    def get_retriever(
        self, 
        retriever_type: str,
        vector_store, 
        top_k: int, 
    ) -> List[str]:
        """
        Get Retriever.
        """
        if retriever_type == "rerank_retriever":
            print(f"Initialize RerankRetriever.")
            retriever = CustomRerankRetriever(
                vector_store=vector_store,
                embed_model=Settings.embed_model,
                similarity_top_k=top_k
            )

        elif retriever_type == "rerank_and_filter_retriever":
            print(f"Initialize RerankAndFilterRetriever.")

            filters = [
                MetadataFilter(
                    key='technology',
                    value=title,
                    operator='==',
                
                )
                for title in ['docker', 'spring-boot']
            ]

            filters = MetadataFilters(filters=filters, condition="or")

            retriever = CustomRerankAndFilterRetriever(
                vector_store=vector_store,
                embed_model=Settings.embed_model,
                similarity_top_k=top_k,
                filters=filters
            )

        elif retriever_type == "auto_merging_retriever":
            print(f"Initialize AutoMergingRetriever.")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            base_retriever = VectorIndexRetriever(
                index=VectorStoreIndex.from_vector_store(vector_store=vector_store),
                similarity_top_k=top_k,
                embed_model=Settings.embed_model
            )
            retriever = AutoMergingRetriever(vector_retriever=base_retriever, storage_context=storage_context)

        else:
            print(f"Initialize BaseRetriever.")
            retriever = CustomBaseRetriever(
                vector_store=vector_store,
                similarity_top_k=top_k,
                embed_model=Settings.embed_model
            )

        return retriever
