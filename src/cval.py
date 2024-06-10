from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from data import Dependency
from ingestion_engine import DataIngestionEngine
from retrieval_engine import RetrievalEngine
from query_engine import QueryEngine
from generator_engine import GeneratorFactory
from typing import List, Optional
from dotenv import load_dotenv
import os


class CVal:
    def __init__(self, model_name: str, temperature: int, env_file_path: str) -> None:
        load_dotenv(dotenv_path=env_file_path)
        self.model_name = model_name
        self.set_settings()
        self.ingestion_engine = DataIngestionEngine()
        self.retrieval_engine = RetrievalEngine()
        self.query_engine = QueryEngine()
        self.generator_engine = GeneratorFactory().get_generator(
            model_name=model_name,
            temperature=temperature
        )

    
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
        # Scrape documents from web
        documents = self.ingestion_engine.scrape(
            query=query, 
            website=website,
            num_websites=num_websites
        )

        # Get VectorStore from RetrievalEngine
        vector_store = self.retrieval_engine.get_vector_store(
            index_name=index_name,
            dimension=dimension,
            metric=metric
        )

        # Add data to vector store
        self.ingestion_engine.index_documents(
            vector_store = vector_store,
            documents=documents
        )


    def validate(
        self, 
        enable_rag: bool,
        dependency: Dependency, 
        index_name: str, 
        retriever_type: str,
        top_k: int
    ) -> List:
        """
        Validate a dependency.
        """
        if not enable_rag:
            messages = [
                {
                    "role": "system", 
                    "content": self.query_engine.get_system_str(dependency=dependency)
                },
                {
                    "role": "user",
                    "content": self.query_engine.get_task_str(dependency=dependency)
                }
            ]

            validation_response = self.generator.generate(messages=messages)

            return validation_response

        vector_store = self.retrieval_engine.get_vector_store(index_name=index_name)
        retriever = self.retrieval_engine.get_retriever(
            retriever_type=retriever_type,
            vector_store=vector_store,
            top_k=top_k
        )

        validation_response = self.query_engine.custom_query(
            retriever=retriever,
            llm=Settings.llm, 
            dependency=dependency
        )

        return validation_response
 


    

