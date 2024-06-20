from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core.schema import NodeWithScore
from llama_index.core.indices.query.schema import QueryBundle
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from data import Dependency, CvalConfig
from ingestion import DataIngestionEngine
from generator import GeneratorFactory
from retriever import RetrieverFactory
from prompt_templates import QUERY_PROMPT, SYSTEM_PROMPT, TASK_PROMPT, DEPENDENCY_STR, FORMAT_STR
from typing import List
from dotenv import load_dotenv
from rich.logging import RichHandler
import os
import logging
import yaml


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()],
)


class CVal:
    def __init__(self, cfg: CvalConfig) -> None:
        self.cfg = cfg
        load_dotenv(dotenv_path=self.cfg.env_file_path)
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.set_embedding_and_inference_model(model_name=self.cfg.model_name)
        
    def set_embedding_and_inference_model(self, model_name: str) -> None:
        if model_name.startswith("gpt"):
            Settings.embed_model = OpenAIEmbedding(api_key=os.getenv(key="OPENAI_KEY"))
            Settings.llm = OpenAI(model=model_name, api_key=os.getenv(key="OPENAI_KEY"))
        elif model_name.startswith("llama"):
            Settings.embed_model = OllamaEmbedding(model_name=model_name)
            Settings.llm = Ollama(model=model_name)
        else:
            raise Exception(f"Model {model_name} not yet supported.")

    def get_vector_store(
        self, 
        index_name: str,
        dimension: int = 1536,
        metric: str = "dotproduct"
    ):       
        """
        Get vector store.
        """
        if index_name not in self.pc.list_indexes().names():
            logging.info(f"Create Index {index_name}.")
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

        index = self.pc.Index(index_name)
        vector_store = PineconeVectorStore(
            pinecone_index=index,
            add_sparse_vector=True
        )

        logging.info(f"Select Index {index_name}.")

        return vector_store
    
    def scrape(
        self, 
        dependency: Dependency
    ) -> None:
        """
        Scrape websites and the repository and index the corresponding data.
        """
        logging.info(f"Start scraping web and Github.")
        ingestion_engine = DataIngestionEngine()

        website_documents = ingestion_engine.docs_from_web(
            dependency=dependency,
            num_websites=self.cfg.num_websites
        )

        repo_documents = ingestion_engine.docs_from_github(
            dependency=dependency
        )
        logging.info(f"Scraping done.")

        documents = website_documents + repo_documents

        vector_store = self.get_vector_store(
            index_name="web-search"
        )
        
        logging.info(f"Start indexing documents in index: web-search.")
        ingestion_engine.index_documents(
            vector_store = vector_store,
            documents=documents
        )
        logging.info(f"Done with indexing documents in index: web-search")

        logging.info("Start indexing documents in index: all.")
        ingestion_engine.index_documents(
            vector_store = self.get_vector_store(index_name="all"),
            documents=documents
        )
        logging.info("Done with indexing documents in index: all.")


    def index_data(
        self, 
        config_file: str
    ) -> None:
        """
        Index data based on a given indexing config.
        """
        if not os.path.exists(config_file):
            raise Exception(f"Indexing config file {config_file} does not exist.")

        with open(config_file, "r", encoding="utf-8") as file:
            indexing_config = yaml.load(file, Loader=yaml.FullLoader)

        ingestion_engine = DataIngestionEngine()

        all_documents = []

        for index_name, config in indexing_config["index"].items():
            vector_store = self.get_vector_store(
                index_name=index_name,
                dimension=config["dimension"],
                metric=config["metric"]
            )    
            
            documents = []

            if config["type"] == "url":
                documents = ingestion_engine.docs_from_urls(url_file=config["path"])

            if config["type"] == "dir":
                documents = ingestion_engine.docs_from_dir(directory=config["path"])

            if not documents:
                raise Exception("Documents could not be loaded.")
            
            logging.info(f"Start indexing documents in index: {index_name}.")
            ingestion_engine.index_documents(
                documents=documents,
                vector_store=vector_store,
                splitting=config["splitting"]
            )
            logging.info(f"Done with indexing documents in index: {index_name}.")

            all_documents += documents

        # index all data into one index
        logging.info(f"Start indexing documents in index: all.")
        ingestion_engine.index_documents(
            documents=all_documents,
            vector_store=self.get_vector_store(
                index_name="all",
                dimension=config["dimension"],
                metric=config["metric"]
            ),
            splitting=config["splitting"]    
        )
        logging.info(f"Done with indexing documents in index: all.")

    def retrieve(
        self, 
        index_name: str, 
        task_str: str
    ) -> List[NodeWithScore]:
        """
        Retrieve relevant nodes from vector store.
        """
        vector_store = self.get_vector_store(index_name=index_name)
        retriever = RetrieverFactory().get_retriever(
            retriever_type=self.cfg.retrieval_type,
            vector_store=vector_store,
            top_k=self.cfg.top_k
        )

        if not retriever:
            raise Exception(f"Retriever type{self.cfg.retrieval_type} not yet supported.")

        retrieved_nodes = retriever.retrieve(QueryBundle(query_str=task_str))

        return retrieved_nodes

    def generate(
        self, 
        system_str: str, 
        context_str: str, 
        task_str: str
    ) -> str:
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
                    "content": f"{task_str}\n\n{FORMAT_STR}"
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

    def query(
        self,
        dependency: Dependency
    ) -> str:
        """
        Validate a given dependency.
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
            dependency_str=DEPENDENCY_STR
        )
    
