from retrieval.retrieval_engine import RetrievalEngine
from generator.generator import GeneratorFactory
from scraper.scraper import Scraper
from data.dependency import Dependency
from typing import Dict, List
from dotenv import load_dotenv
from rich.logging import RichHandler
import tempfile
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()],
)


class CVal:
    def __init__(self, config: Dict) -> None:
        self.config = config
        load_dotenv()

    def retrieve(self, query: str) -> List:
        """
        Retrieve relevant text from vector store.
        """
        retrieval_engine = RetrievalEngine(embed_model_name=self.config.embed_model_name)

        if self.config.index_name == "web-search":
            logging.info("Start scraping new documents.")
            scraper = Scraper()
            with tempfile.TemporaryDirectory() as temp_dir:
                scraper.scrape(output_dir=temp_dir, query=query, num_documents=self.config.num_documents), 
                retrieval_engine.add_documents(
                    index_name=self.config.index_name, 
                    document_dir=temp_dir,
                    dimension=self.config.dimension,
                    metric=self.config.metric
                )

        context_list = retrieval_engine.retrieve_context(
            query=query,
            index_name=self.config.index_name,
            top_k=self.config.top_k
        )

        return context_list


    def generate(self, dependency: Dependency, context_str: List) -> str:
        """
        Generate answer from context.
        """
        generator = GeneratorFactory().get_generator(model_name=self.config.model_name)

        messages = []

        response = generator.generate(messages=messages)

        return response.choices[0].message.content

    
    def validate(self, dependency: Dependency) -> None:
        """
        Validate the given dependency.
        """

        context_str = self.retrieve(dependency=dependency)
        completion = self.generate(
            dependency=dependency,
            context_str=context_str
        )
        return completion