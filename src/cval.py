from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModelType
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings, Document
from llama_index.core.schema import NodeWithScore
from data import Dependency
from ingestion import IngestionEngine
from generator import GeneratorFactory
from retrieval import RetrievalEngine
from weaviate import WeaviateClient
import weaviate
from prompt_templates import QUERY_PROMPT, SYSTEM_PROMPT, TASK_PROMPT, FORMAT_STR
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


class CVal:
    def __init__(self, config_file: str, env_file: str) -> None:
        self.config = self._load_config(config_file=config_file)
        load_dotenv(dotenv_path=env_file)

        self._set_embed_model()
        self._set_inference_model()

        self._weaviate_client = self._get_weaviate_client()

        self._ingestion_engine = IngestionEngine(
            weaviate_client=self._weaviate_client
        )
        self._retrieval_engine = RetrievalEngine(
            weaviate_client=self._weaviate_client,
            search_strategy=self.config["search_strategy"],
            distance_metrics=self.config["distance_metric"],
            top_k=self.config["top_k"],
            top_n=self.config["top_n"],
            num_queries=self.config["num_queries"],
            alpha=self.config["alpha"]
        )

    def _load_config(self, config_file: str) -> Dict:
        """
        Load configuration from TOML file.
        """
        if not config_file.endswith(".toml"):
            raise Exception("Config file has to be a TOML file.")
        
        with open(config_file, "r", encoding="utf-8") as f:
            config = toml.load(f)
        
        return config

    def _get_weaviate_client(self) -> WeaviateClient:
        """
        Create weaviate client.
        """
        weaviate_client = weaviate.connect_to_custom(
            http_host=self.config["weaviate_host"],
            http_port=self.config["weaviate_http_port"],
            http_secure=False,
            grpc_host=self.config["weaviate_host"],
            grpc_port=self.config["weaviate_grpc_port"],
            grpc_secure=False,
        )
        return weaviate_client

    def _set_embed_model(self) -> None:
        """
        Set the embedding model.
        """
        Settings.embed_model = None

        if self.config["embed_model"].startswith("llama"):
            Settings.embed_model = OllamaEmbedding(model_name=self.config["embed_model"])
        
        if self.config["embed_model"].startswith("gpt"):
            Settings.embed_model = OpenAIEmbedding(
                api_key=os.getenv(key="OPENAI_KEY"),
                model=OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002
            )

    def _set_inference_model(self) -> None:
        """
        Set inference model.
        """
        Settings.llm = None

        if self.config["inference_model"].startswith("llama"):
            Settings.llm = Settings.llm = Ollama(model=self.config["inference_model"])
        
        if self.config["inference_model"].startswith("gpt"):
            Settings.llm = OpenAI(
                api_key=os.getenv(key="OPENAI_KEY"),
                model=self.config["inference_model"]
            )

    
    def _scrape_web(self, dependency: Dependency) -> List[Document]:
        """
        Scrape the web for relevant context.
        """
        ingestion_engine = IngestionEngine(
            weaviate_host=self.config["weaviate_host"],
            weaviate_http_port=self.config["weaviate_http_port"],
            weaviate_grpc_port=self.config["weaviate_grpc_port"]
        )

        logging.info(f"Srape the web.")
        documents = ingestion_engine.docs_from_web(
            dependency=dependency,
            num_websites=self.config["num_websites"]
        )
        logging.info(f"Scraping done.")

        logging.info(f"Index data from web.")
        ingestion_engine.index(
            index_name="WebSearch",
            documents=documents,
            splitting=self.config["splitting"],
            delete_index=False
        )

        ingestion_engine.index(
            index_name="All",
            documents=documents,
            splitting=self.config["splitting"],
            delete_index=False
        )
        logging.info(f"Indexing done.")


    def init_rag(self) -> None:
        """
        Enable RAG.
        """
        all_docs = []

        web_docs = self._ingestion_engine.docs_from_urls(urls=self.config["urls"])

        self._ingestion_engine.index(
            index_name="TechDocs",
            documents=web_docs,
            splitting=self.config["splitting"],
            extractors=self.config["extractors"]
        )

        so_docs = self._ingestion_engine.docs_from_dir(dir=self.config["so_post_dir"])

        self._ingestion_engine.index(
            index_name="SoPosts",
            documents=so_docs,
            splitting=self.config["splitting"],
            extractors=self.config["extractors"]
        )

        all_docs += web_docs + so_docs

        self._ingestion_engine.index(
            index_name="All",
            documents=all_docs,
            splitting=self.config["splitting"],
            extractors=self.config["extractors"]
        )

    
    def retrieve(
        self, 
        index_name: str, 
        task_str: str
    ) -> List[NodeWithScore]:
        """
        Retrieve relevant context.
        """
        retrieved_nodes = self._retrieval_engine.retrieve(
            index_name=index_name,
            query_str=task_str,
            rerank=self.config["rerank"]
        )

        return retrieved_nodes

    def generate(
        self, 
        system_str: str,
        task_str: str
    ) -> str:
        """
        Generate answer from context.
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
            model_name=self.config["inference_model"],
            temperature=self.config["temperature"]
        )
        response = generator.generate(messages=messages)

        return response
        

    def generate(
        self, 
        system_str: str,
        context_str: str,
        task_str: str
    ) -> str:
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


    def query(self, dependency: Dependency) -> str:
        """
        Query RAG.
        """
        # create system_str
        system_str = SYSTEM_PROMPT.format(
            project=dependency.project,
        )
        
        # create task str
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
        
        # query llm with context from RAG
        if self.config["with_rag"]:
            retrieved_nodes = self.retrieve(
                index_name=self.config["target_index_name"],
                task_str=task_str
            )


            return retrieved_nodes
            #context_str = "\n\n".join([n.node.get_content() for n in retrieved_nodes])

            #print("Context: ", context_str)

            #response = self.generate(
            #    system_str=system_str,
            #    context_str=context_str,
            #    task_str=task_str,
            #)

        # query llm without context from llm
        #else:
        #    response = self.generate(
        #        system_str=system_str,
        #        task_str=task_str,
        #    )

        #return response
    
    
