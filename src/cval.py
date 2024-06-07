from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings, get_response_synthesizer
from data import ValidationResponse, Dependency
from ingestion_engine import DataIngestionEngine
from retrieval_engine import RetrievalEngine
from llama_index.core.query_engine import RetrieverQueryEngine
from generator import GeneratorFactory
from prompt_templates import SYSTEM_PROMPT, USER_PROMPT, DEPENDENCY_PROMPT
from typing import List, Optional
from dotenv import load_dotenv
import os


class CVal:
    def __init__(self, model_name: str, env_file_path: str) -> None:
        load_dotenv(dotenv_path=env_file_path)
        self.model_name = model_name
        self.set_settings()
        self.ingestion_engine = DataIngestionEngine()
        self.retrieval_engine = RetrievalEngine()

    
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
                    "content": SYSTEM_PROMPT.format(
                        dependency.option_technology,
                        dependency.dependent_option_technology,
                        dependency.project
                    )
                },
                {
                    "role": "user",
                    "content": USER_PROMPT.format(DEPENDENCY_PROMPT.format(
                        dependency.dependency_type,
                        dependency.project,
                        dependency.option_name,
                        dependency.option_value,
                        dependency.option_type,
                        dependency.option_file,
                        dependency.option_technology,
                        dependency.dependent_option_name,
                        dependency.dependent_option_value,
                        dependency.dependency_type,
                        dependency.dependent_option_file,
                        dependency.dependent_option_technology,
                        dependency.dependency_category
                    ))
                }
            ]

            generator = GeneratorFactory().get_generator(
                model_name=self.model_name,
                temperature=0.0
            )
            
            validation_response = generator.generate(messages=messages)

            return validation_response


        vector_store = self.retrieval_engine.get_vector_store(index_name=index_name)
        retriever = self.retrieval_engine.get_retriever(
            retriever_type=retriever_type,
            vector_store=vector_store,
            top_k=top_k
        )

        #query_bundle = QueryBundle(query)
        #retrieved_nodes = retriever.retrieve(query_bundle)
        #for node in retrieved_nodes:
        #    print(node)

        response_synthesizer = get_response_synthesizer()

        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            llm=Settings.llm,
            output_cls=ValidationResponse,
            response_mode="compact",
        )

        #query_engine = RetrieverQueryEngine(
        #    retriever=retriever,
        #    response_synthesizer=response_synthesizer,
        #    node_postprocessors=[
        #        SimilarityPostprocessor(similarity_cutoff=0.7)
        #    ]
        #)

        validation_response = query_engine.query(
            str_or_query_bundle=self._create_query(dependency=dependency)
        )        

        return validation_response
 
    def _create_query(self, dependency: Dependency) -> str:
        return USER_PROMPT.format(DEPENDENCY_PROMPT.format(
            dependency.dependency_type,
            dependency.project,
            dependency.option_name,
            dependency.option_value,
            dependency.option_type,
            dependency.option_file,
            dependency.option_technology,
            dependency.dependent_option_name,
            dependency.dependent_option_value,
            dependency.dependency_type,
            dependency.dependent_option_file,
            dependency.dependent_option_technology,
            dependency.dependency_category
        ))


    

