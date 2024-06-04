from retrieval.retrieval_engine import RetrievalEngine
from prompting.prompt_manager import PromptManager
from generator.generator import GeneratorFactory
from scraper.scraper import Scraper
from data.dependency import Dependency
from typing import Dict, List
from dotenv import load_dotenv


class CVal:
    def __init__(self, config: Dict) -> None:
        self.config = config
        load_dotenv()

    def retrieve(self, dependency: Dependency) -> List:
        """
        Retrieve relevant text from vector store.
        """
        retrieval_engine = RetrievalEngine(embed_model_name=self.config.embed_model)

        if self.config.mode == "live":
            raise NotImplementedError()
        else:
            index = retrieval_engine.get_index(index_name=self.config.index_name)
            vector_store = retrieval_engine.get_vector_store(index=index)
            raise NotImplementedError()


    def generate_completion(self, dependency: Dependency, context_str: List) -> str:
        """
        Generate answer from context.
        """
        generator = GeneratorFactory().get_generator(model_name=self.config.model_name)

        messages = PromptManager().create_prompt(
            dependency=dependency, 
            context_str=context_str
        )

        response = generator.generate(messages=messages)

        return response.choices[0].message.content

    
    def validate(self, dependency: Dependency) -> None:
        """
        Validate the given dependency.
        """

        context_str = self.retrieve(dependency=dependency)
        completion = self.generate_completion(
            dependency=dependency,
            context_str=context_str
        )
        return completion