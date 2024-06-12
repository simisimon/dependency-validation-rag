from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings, SimpleDirectoryReader
from data import Dependency, CvalConfig
from ingestion_engine import DataIngestionEngine
from retrieval_engine import RetrievalEngine
from query_engine import QueryEngine
from generator_engine import GeneratorFactory
from typing import List, Dict
from dotenv import load_dotenv
from rich.logging import RichHandler
import os
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()],
)


class CVal:
    def __init__(self, cfg: CvalConfig) -> None:
        self.cfg = cfg
        load_dotenv(dotenv_path=self.cfg.env_file_path)
        self.set_settings()
    
    def set_settings(self) -> None:
        if self.cfg.model_name.startswith("gpt"):
            Settings.embed_model = OpenAIEmbedding(api_key=os.getenv(key="OPENAI_KEY"))
            Settings.llm = OpenAI(model=self.cfg.model_name, api_key=os.getenv(key="OPENAI_KEY"))
        elif self.cfg.model_name.startswith("llama"):
            Settings.embed_model = OllamaEmbedding(model_name=self.cfg.model_name)
            Settings.llm = Ollama(model=self.model_name)
        else:
            raise Exception(f"Model {self.cfg.model_name} not yet supported.")
            
    def scrape(self, dependency: Dependency) -> None:
        """
        Scrape websites from the web and index the corresponding data.
        """
        logging.info(f"Start scraping.")
        ingestion_engine = DataIngestionEngine()

        website_documents = ingestion_engine.scrape_websites(
            dependency=dependency,
            num_websites=self.cfg.num_websites
        )

        repo_documents = ingestion_engine.scrape_repositories(
            dependency=dependency
        )
        logging.info(f"Scraping done.")

        documents = website_documents + repo_documents

        vector_store = RetrievalEngine().get_vector_store(
            index_name="web-search",
            dimension=1536,
            metric="cosine"
        )
        
        logging.info(f"Start indexing {len(documents)} documents.")
        ingestion_engine.index_documents(
            vector_store = vector_store,
            documents=documents
        )
        logging.info(f"indexing done.")


    def validate(self, dependency: Dependency) -> List:
        """
        Validate a dependency.
        """
        if not self.cfg.enable_rag:
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

            generator = GeneratorFactory().get_generator(
                model_name=self.cfg.model_name,
                temperature=self.cfg.temperature
            )
            response = generator.generate(messages=messages)

            return response

        retrieval_engine = RetrievalEngine()
        vector_store = retrieval_engine.get_vector_store(index_name=self.cfg.index_name)
        retriever = retrieval_engine.get_retriever(
            retriever_type=self.cfg.retrieval_type,
            vector_store=vector_store,
            top_k=self.cfg.top_k
        )

        response = QueryEngine().custom_query(
            retriever=retriever,
            llm=Settings.llm, 
            dependency=dependency
        )

        return response   

