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
from prompt_templates import QUERY_PROMPT, SYSTEM_PROMPT, TASK_PROMPT, DEPENDENCY_STR, FORMAT_STR
from typing import List, Dict
from dotenv import load_dotenv
from rich.logging import RichHandler
from util import is_index_empty
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

        self.generator = GeneratorFactory().get_generator(
            model_name=self.config["inference_model"],
            temperature=self.config["temperature"]
        )

        logging.info(f"CVal initialized.")

    def _scrape(self, dependency: Dependency) -> None:
        """
        Scrape websites and the repository and index the corresponding data.
        """
        # create ingestion engine
        logging.info(f"Scraping web.")
        web_docs = self.ingestion_engine.docs_from_web(
            dependency=dependency,
            num_websites=self.config["num_websites"]
        )
    
        logging.info(f"Indexing data into 'web-search'.")
        self.ingestion_engine.index_documents(
            index_name="web-search",
            documents=web_docs,
            delete_index=False
        )
    
    def retrieve(self, index_name: str, task_str: str) -> List[NodeWithScore]:
        """
        Retrieve relevant context.
        """
        retrieved_nodes = self.retrieval_engine.retrieve(
            index_name=index_name,
            query_str=task_str
        )

        return retrieved_nodes

    def generate(self, messages: List) -> str:
        """
        Generate answer.
        """
        response = self.generator.generate(messages=messages)

        return response

    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    def query(self, dependency: Dependency, index_name: str) -> Response:
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

        # create query string with context
        if self.config["with_rag"]:

            if index_name == "web-search":
                self._scrape(dependency=dependency)   

            retrieved_nodes = self.retrieve(
                index_name=index_name,
                task_str=task_str
            )

            context_str = "\n\n".join([source_node.node.get_content() for source_node in retrieved_nodes])

            query_str = QUERY_PROMPT.format(
                context_str=context_str, 
                task_str=task_str,
                format_str=FORMAT_STR
            ) 

        # create query string without context
        else:
            query_str = f"{task_str}\n\n{FORMAT_STR}"
        
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
            input="\n".join([x["content"] for x in messages]),
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
            if is_index_empty(index=pinecone_client.Index(name="tech-docs")):
                logging.info("Index data into 'tech-docs'.")
                docs = ingestion_engine.docs_from_urls(urls=data_config["urls"])
                ingestion_engine.index_documents(
                    index_name="tech-docs",
                    documents=docs,
                    delete_index=True
                )

        if "so-posts" not in pinecone_client.list_indexes().names():
            if is_index_empty(index=pinecone_client.Index(name="so-posts")):
                logging.info("Index data into 'so-posts'.")
                docs = ingestion_engine.docs_from_dir(data_dir=data_config["data_dir"])
                ingestion_engine.index_documents(
                    index_name="so-posts",
                    documents=docs,
                    delete_index=True
                )

        if "github" not in pinecone_client.list_indexes().names():
            if is_index_empty(index=pinecone_client.Index(name="github")):
                logging.info("Index data into 'github'.")
                docs = [ingestion_engine.docs_from_github(project_name=project_name) for project_name in data_config["github"]]
                ingestion_engine.index_documents(
                    index_name="github",
                    documents=docs,
                    delete_index=True
                )

        # return cval instance
        return CVal(config=cval_config)


