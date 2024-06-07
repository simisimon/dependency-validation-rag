from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index_client import MetadataFilter, MetadataFilters
from pinecone import Pinecone, ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever, AutoMergingRetriever
from data import Dependency
from data_ingestion import DataIngestionEngine
from retrieval import CustomRerankRetriever, CustomRerankAndFilterRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.query.schema import QueryBundle
from typing import Dict, List, Optional
from dotenv import load_dotenv
from llama_index.core.response.notebook_utils import display_source_node
import os


class CVal:
    def __init__(self, model_name: str, env_file_path: str) -> None:
        load_dotenv(dotenv_path=env_file_path)
        self.model_name = model_name
        self.vector_db_instance = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.set_settings()


    def set_settings(self) -> None:
        if self.model_name.startswith("gpt"):
            Settings.embed_model = OpenAIEmbedding(api_key=os.getenv(key="OPENAI_KEY"))
            Settings.llm = OpenAI(model=self.model_name, api_key=os.getenv(key="OPENAI_KEY"))
        elif self.model_name.startswith("llama"):
            Settings.embed_model = OllamaEmbedding(model_name=self.model_name)
            Settings.llm = Ollama(model=self.model_name)
        else:
            raise Exception(f"Model {self.model_name} not yet supported.")
            

    def scrape_data(
        self, 
        query: str, 
        website: Optional[str], 
        num_websites: int,
        dimension: int = 1536,
        metric: str = "cosine",
        index_name: str = "web-search"
    ) -> None:
        """
        Scrape websites from the web and index the corresponding data.
        """
        vector_store = self._get_vector_store(
            index_name=index_name,
            dimension=dimension,
            metric=metric
        )

        ingestion_engine = DataIngestionEngine()

        documents = ingestion_engine.scrape(
            query=query, 
            website=website,
            num_websites=num_websites
        )

        ingestion_engine.index_documents(
            vector_store = vector_store,
            documents=documents
        )      


    def validate(
        self, 
        query: str, 
        index_name: str, 
        retriever_type: str,
        top_k: int
    ) -> List:
        """
        Validate dependencies.
        """
        vector_store = self._get_vector_store(index_name=index_name)

        retrieved_nodes = self.get_retrieved_nodes(
            query=query,
            retriever_type=retriever_type,
            vector_store=vector_store,
            top_k=top_k
        ) 

        for node in retrieved_nodes:
            print(node)


        #response_synthesizer = get_response_synthesizer()

        #query_engine = RetrieverQueryEngine(
        #    retriever=retriever,
        #    response_synthesizer=response_synthesizer,
        #    node_postprocessors=[
        #        SimilarityPostprocessor(similarity_cutoff=0.7)
        #    ]
        #)

        #response = query_engine.query(
        #    str_or_query_bundle=query
        #)        

        #print(response)


    def _get_vector_store(
        self, 
        index_name: str,
        dimension: int = 1536,
        metric: str = "cosine"
    ):
        """
        Create and return vector store.
        """
        if index_name not in self.vector_db_instance.list_indexes().names():
            self.vector_db_instance.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

        index = self.vector_db_instance.Index(index_name)
        vector_store = PineconeVectorStore(pinecone_index=index)

        return vector_store


    def get_retrieved_nodes(
        self,
        query: str, 
        retriever_type: str,
        vector_store, 
        top_k: int, 
    ) -> List[str]:
        """
        Retrieve relevant nodes from the vector database.
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
            retriever = VectorIndexRetriever(
                index=VectorStoreIndex.from_vector_store(vector_store=vector_store),
                similarity_top_k=top_k,
                embed_model=Settings.embed_model
            )


        query_bundle = QueryBundle(query)
        retrieved_nodes = retriever.retrieve(query_bundle)

        return retrieved_nodes