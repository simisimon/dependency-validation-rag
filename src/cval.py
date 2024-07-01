from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModelType
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core.schema import NodeWithScore
from pinecone import Pinecone
from ingestion import IngestionEngine
from generator import GeneratorFactory
from retrieval import RetrievalEngine
from data import Dependency, Response
from prompt_settings import PrompSettingsFactory
from typing import List, Dict
from dotenv import load_dotenv
from rich.logging import RichHandler
import backoff
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
            model=inference_model_name,
            request_timeout=90.0
    )

    if not Settings.llm:
        raise Exception("Inference model has to be set.")




class CVal:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self._pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        self.retrieval_engine = RetrievalEngine(
            pinecone_client=self._pinecone_client,
            with_rewriting=self.config["with_rewriting"],
            rerank=self.config["rerank"],
            top_k=self.config["top_k"],
            top_n=self.config["top_n"],
            alpha=self.config["alpha"]
        )

        self.ingestion_engine = IngestionEngine(
            pinecone_client=self._pinecone_client,
            splitting=self.config["splitting"],
            extractors=self.config["extractors"]
        )

        self.generator = GeneratorFactory().get_generator(
            model_name=self.config["inference_model"],
            temperature=self.config["temperature"]
        )

        self.prompt_settings = PrompSettingsFactory.get_prompt_settings(
            tool_name=self.config["tool_name"]
        )


        logging.info(f"CVal initialized.")

    def _scrape(self, dependency: Dependency) -> None:
        """
        Scrape websites and the repository and index the corresponding data.
        """
        # create ingestion engine
        logging.info(f"Scraping web.")
        web_docs = self.ingestion_engine.docs_from_web(
            query_str=self.prompt_settings.get_retrieval_prompt(
                dependency=dependency
            ),
            num_websites=self.config["num_websites"]
        )
    
        logging.info(f"Indexing data into 'web-search'.")
        self.ingestion_engine.index_documents(
            index_name="web-search",
            documents=web_docs,
            delete_index=False
        )
    
    def retrieve(self, index_name: str, retrieval_str: str) -> List[NodeWithScore]:
        """
        Retrieve relevant context.
        """
        retrieved_nodes = self.retrieval_engine.retrieve(
            index_name=index_name,
            query_str=retrieval_str
        )

        return retrieved_nodes

    def generate(self, messages: List) -> str:
        """
        Generate answer.
        """
        logging.info(f"Query {self.generator.model_name}")
        response = self.generator.generate(messages=messages)
        return response

    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    def query(self, dependency: Dependency, index_name: str) -> Response:
        """
        Validate a given dependency.
        """
        # create system prompt
        system_str = self.prompt_settings.get_system_str(
            dependency=dependency
        )

        # create task prompt
        task_str = self.prompt_settings.get_task_str(
            dependency=dependency
        )

        # create query string with context
        if self.config["with_rag"]:
            logging.info("Query LLM with context.")

            if index_name == "web-search":
                self._scrape(dependency=dependency)   

            
            if self.config["with_rewriting"]:
                logging.info("Rewrite query.")
                retrieval_str = self.prompt_settings.get_retrieval_prompt(
                    dependency=dependency
                )
            else:
                retrieval_str = task_str

            retrieved_nodes = self.retrieve(
                index_name=index_name,
                retrieval_str=retrieval_str
            )

            context_str = "\n\n".join([source_node.node.get_content() for source_node in retrieved_nodes])

            query_str = self.prompt_settings.query_prompt.format(
                context_str=context_str, 
                task_str=task_str,
                format_str=self.prompt_settings.get_format_prompt()
            ) 

        # create query string without context
        else:
            logging.info("Query LLM without context.")
            query_str = f"{task_str}\n\n{self.prompt_settings.get_format_prompt()}"
            retrieved_nodes = []
        
        messages = [
            {
                "role": "system", 
                "content": system_str
            },
            {
                "role": "user",
                "content": query_str
            }
        ]

        response = self.generate(messages=messages)
        return Response(
            input=f"{task_str}\n\n{self.prompt_settings.get_format_prompt()}",
            response=response,
            source_nodes=retrieved_nodes
        )
    

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

        # create ingestion engine
        ingestion_engine = IngestionEngine(
            pinecone_client=pinecone_client,
            splitting=cval_config["splitting"],
            extractors=cval_config["extractors"]
        )

        if "tech-docs" not in pinecone_client.list_indexes().names():
            logging.info("Index data into 'tech-docs'.")
            docs = ingestion_engine.docs_from_urls(urls=data_config["urls"])
            ingestion_engine.index_documents(
                index_name="tech-docs",
                documents=docs,
                delete_index=True
            )

        if "so-posts" not in pinecone_client.list_indexes().names():
            logging.info("Index data into 'so-posts'.")
            docs = ingestion_engine.docs_from_dir(data_dir=data_config["data_dir"])
            ingestion_engine.index_documents(
                index_name="so-posts",
                documents=docs,
                delete_index=True
            )

        if "github" not in pinecone_client.list_indexes().names():
            logging.info("Index data into 'github'.")
            docs = []
            for project_name in data_config["github"]:
                docs += ingestion_engine.docs_from_github(project_name=project_name)

            ingestion_engine.index_documents(
                index_name="github",
                documents=docs,
                delete_index=True
            )

        # return cval instance
        return CVal(config=cval_config)


