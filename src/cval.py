from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModelType
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core.schema import NodeWithScore
from llama_index.core.indices.query.schema import QueryBundle
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from data import Dependency, CvalConfig
from ingestion import IngestionEngine
from generator import GeneratorFactory
from retrieval import RetrievalEngine
from prompt_templates import QUERY_PROMPT, SYSTEM_PROMPT, TASK_PROMPT, DEPENDENCY_STR, FORMAT_STR
from typing import List, Dict
from dotenv import load_dotenv
from rich.logging import RichHandler
import os
import logging
import toml


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()],
)


def load_config(config_file: str) -> Dict:
    """
    Load config from TOML file.
    """
    if not config_file.endswith(".toml"):
            raise Exception("Config file has to be a TOML file.")
        
    with open(config_file, "r", encoding="utf-8") as f:
        config = toml.load(f)
        
    return config


def set_embedding(embed_model_name: str) -> None:
    """
    Set embedding model.
    """
    if embed_model_name.startswith("openai"):
        Settings.embed_model = OpenAIEmbedding(
            api_key=os.getenv(key="OPENAI_KEY"),
            model=OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002
        )

    if embed_model_name.startswith("llama"):
        Settings.embed_model = OllamaEmbedding(model_name=embed_model_name)

    if not Settings.embed_model:
        raise Exception("Embedding model has to be set.")


def set_llm(inference_model_name: str) -> None: 
    """
    Set inference model.
    """
    if inference_model_name.startswith("gpt"):
        Settings.llm = OpenAI(
            model=inference_model_name, 
            api_key=os.getenv(key="OPENAI_KEY")
        )
        
    if inference_model_name.startswith("llama"):
        Settings.llm = Ollama(
            model_name=inference_model_name
    )

    if not Settings.llm:
        raise Exception("Inference model has to be set.")


class CVal:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self._pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
        self.retrieval_engine = RetrievalEngine(
            pinecone_client=self._pinecone_client,
            rerank=self.config["rerank"],
            top_k=self.config["top_k"],
            top_n=self.config["top_n"],
            num_queries=self.config["num_queries"],
            alpha=self.config["alpha"]
        )

        self.ingestion_engine = IngestionEngine(
            pinecone_client=self._pinecone_client,
            splitting=self.config["splitting"],
            extractors=self.config["extractors"]
        )


    def scrape(
        self, 
        dependency: Dependency
    ) -> None:
        """
        Scrape websites and the repository and index the corresponding data.
        """
        # create ingestion engine
        logging.info(f"Scraping web.")
        web_docs = self.ingestion_engine.docs_from_web(
            dependency=dependency,
            num_websites=self.config["num_websites"]
        )
        logging.info(f"Scraping web done.")

        logging.info(f"Scraping GitHub.")
        repo_docs= self.ingestion_engine.docs_from_github(
            project_name=dependency.project
        )
        logging.info(f"Scraping GitHub done.")

        docs = web_docs + repo_docs

        logging.info(f"Indexing data from web and Github.")
        self.ingestion_engine.index_documents(
            index_name="web-search",
            documents=docs,
            delete_index=False
        )

        self.ingestion_engine.index(
            index_name="all",
            documents=docs,
            splitting=self.config["splitting"],
            delete_index=False
        )
        logging.info(f"Indexing data from web and Github done.")
    

    def retrieve(self, index_name: str, task_str: str) -> List[NodeWithScore]:
        """
        Retrieve relevant context.
        """
        retrieved_nodes = self.retrieval_engine.retrieve(
            index_name=index_name,
            query_str=task_str
        )

        return retrieved_nodes

    def generate(self, system_str: str, task_str: str) -> str:
        """
        Generate answer without context.
        """
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
            model_name=self.config["inference"],
            temperature=self.config["temperature"]
        )
        response = generator.generate(messages=messages)

        return response

    def generate(self, system_str: str, context_str: str, task_str: str) -> str:
        """
        Generate answer with context.
        """
        query_str = QUERY_PROMPT.format(
            system_str=system_str,
            context_str=context_str, 
            task_str=task_str,
            format_str=FORMAT_STR
        )   

        response = Settings.llm.complete(
            prompt=query_str,
            temperature=self.config["temperature"]
        )

        return response


    def query(self, dependency: Dependency, index_name: str) -> str:
        """
        Validate a given dependency.
        """
        # create system prompt
        system_str = SYSTEM_PROMPT.format(
            project=dependency.project,
            dependency_str=DEPENDENCY_STR
        )

        # create task prompt
        task_str = TASK_PROMPT.format(
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

        # generate answer with context
        if self.config["with_rag"]:
            retrieved_nodes = self.retrieve(
                index_name=index_name,
                task_str=task_str
            )

            context_str = "\n\n".join([n.get_content() for n in retrieved_nodes])
    
            response = self.generate(
                system_str=system_str,
                context_str=context_str,
                task_str=task_str,
            )

            return response
            
        # generate answer without context
        else:
            response = self.generate(
                system_str=system_str,
                task_str=task_str,
            )
            return response
    

    @staticmethod
    def init(config_file: str, env_file: str) -> None:
        """
        Initialize CVal.
        """ 
    
        

        # load env variables from .env file
        load_dotenv(dotenv_path=env_file)

        # load config from TOML file
        config = load_config(config_file=config_file)
        cval_config = config["cval"]
        data_config = config["data"]

        # set embed and inference model
        set_embedding(embed_model_name=cval_config["embed_model"])
        set_llm(inference_model_name=cval_config["inference_model"])

        # create pinecone client
        pinecone_client = Pinecone(api_key=os.getenv(key="PINECONE_API_KEY"))

        if all(index in pinecone_client.list_indexes().names() for index in ["tech-docs", "so-posts", "all"]):
            logging.info("All indexes already exist.")
            # return cval instance
            return CVal(config=cval_config)

        # create ingestion engine
        ingestion_engine = IngestionEngine(
            pinecone_client=pinecone_client,
            splitting=cval_config["splitting"],
            extractors=cval_config["extractors"]
        )

        all_docs = []

        logging.info("Index data into 'tech-docs'.")
        docs = ingestion_engine.docs_from_urls(urls=data_config["urls"])
        ingestion_engine.index_documents(
            index_name="tech-docs",
            documents=docs,
            delete_index=True
        ) 
        all_docs += docs

        logging.info("Index data into 'so-docs'.")
        docs = ingestion_engine.docs_from_dir(data_dir=data_config["data_dir"])
        ingestion_engine.index_documents(
            index_name="so-posts",
            documents=docs,
            delete_index=True
        )
        all_docs += docs

        logging.info("Index data into 'all'.")
        ingestion_engine.index_documents(
            index_name="all",
            documents=all_docs,
            delete_index=True
        )

        # return cval instance
        return CVal(config=cval_config)


