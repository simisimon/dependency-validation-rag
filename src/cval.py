from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core.schema import NodeWithScore
from llama_index.core.indices.query.schema import QueryBundle
from data import Dependency, CvalConfig
from ingestion_engine import DataIngestionEngine
from retrieval_engine import RetrievalEngine
from generator_engine import GeneratorFactory
from prompt_templates import QUERY_PROMPT, SYSTEM_PROMPT, TASK_PROMPT, VALUE_EQUALITY_DEFINITION_STR, FORMAT_STR
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


    def retrieve(self, index_name, task_str: str) -> List[NodeWithScore]:
        """
        Retrieve relevant nodes from vector store.
        """
        retrieval_engine = RetrievalEngine()
        vector_store = retrieval_engine.get_vector_store(index_name=index_name)
        retriever = retrieval_engine.get_retriever(
            retriever_type=self.cfg.retrieval_type,
            vector_store=vector_store,
            top_k=self.cfg.top_k
        )

        retrieved_nodes = retriever.retrieve(QueryBundle(query_str=task_str))

        return retrieved_nodes

    def generate(self, system_str: str, context_str: str, task_str: str) -> str:
        """
        Generate answer from context.
        """
        if not self.cfg.enable_rag:
            messages = [
                {
                    "role": "system", 
                    "content": system_str
                },
                {
                    "role": "user",
                    "content": task_str
                }
            ]

            generator = GeneratorFactory().get_generator(
                model_name=self.cfg.model_name,
                temperature=self.cfg.temperature
            )
            response = generator.generate(messages=messages)

            return response

        else:
            query_str = QUERY_PROMPT.format(
                system_str=system_str,
                context_str=context_str, 
                task_str=task_str,
                format_str=FORMAT_STR
            )   

            response = Settings.llm.complete(
                prompt=query_str,
                temperature=self.cfg.temperature
            )

            return response

    def query(self, dependency: Dependency) -> List:
        """
        Validate a dependency.
        """
        system_str = self._get_system_prompt(dependency=dependency)
        task_str = self._get_task_prompt(dependency=dependency)

        retrieved_nodes = self.retrieve(
            index_name=self.cfg.index_name,
            task_str=task_str
        )

        context_str = "\n\n".join([n.node.get_content() for n in retrieved_nodes])
 
        response = self.generate(
            system_str=system_str,
            context_str=context_str,
            task_str=task_str,
        )

        return response

    def _get_task_prompt(self, dependency: Dependency) -> str:
        return TASK_PROMPT.format(
            nameA=dependency.option_name,
            typeA=dependency.option_type,
            valueA=dependency.option_value,
            fileA=dependency.option_file,
            technologyA=dependency.option_technology,
            nameB=dependency.dependent_option_name,
            typeB=dependency.dependent_option_type,
            valueB=dependency.dependent_option_value,
            fileB=dependency.dependent_option_file,
            technologyB=dependency.dependent_option_technology
        )
    
    def _get_system_prompt(self, dependency: Dependency) -> str:
        return SYSTEM_PROMPT.format(
            project=dependency.project,
            definition_str=VALUE_EQUALITY_DEFINITION_STR
        )
    
