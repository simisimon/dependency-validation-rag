from llama_index.core import Settings, get_response_synthesizer
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.schema import QueryBundle
from data import ValidationResponse, Dependency
from prompt_templates import QUERY_PROMPT, RETRIEVAL_STR, SYSTEM_STR, TASK_STR
from rich.logging import RichHandler
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()],
)


class QueryEngine(CustomQueryEngine):
    def __init__(self):
        logging.info(f"Custom QueryEngine initialized.")

    def custom_query(
        self, 
        retriever: BaseRetriever, 
        llm: BaseLLM,
        dependency: Dependency, 
        temperature: float = 0.0
    ):
        nodes = retriever.retrieve(
            self.get_retrieval_str(dependency=dependency)
        )
        context_str = "\n\n".join([n.node.get_content() for n in nodes])
        response = llm.complete(
            QUERY_PROMPT.format(
                system_str=self.get_system_str(dependency=dependency),
                context_str=context_str, 
                task_str=self.get_task_str(dependency=dependency),
                temperature=temperature
            )
        )
        return response

    def get_retrieval_str(self, dependency: Dependency) -> str:
        return RETRIEVAL_STR.format(
            project=dependency.project,
            nameA=dependency.option_name,
            valueA=dependency.option_value,
            typeA=dependency.option_type,
            fileA=dependency.option_file,
            technologyA=dependency.option_technology,
            nameB=dependency.dependent_option_name,
            valueB=dependency.dependent_option_value,
            typeB=dependency.dependent_option_type,
            fileB=dependency.dependent_option_file,
            technologyB=dependency.dependent_option_technology
        )

    def get_system_str(self, dependency: Dependency) -> str:
        return SYSTEM_STR.format(
            technologyA=dependency.option_technology,
            technologyB=dependency.dependent_option_technology,
            project=dependency.project
        )
    
    def get_task_str(self, dependency: Dependency) -> str:
        return TASK_STR.format(
            project=dependency.project,
            nameA=dependency.option_name,
            valueA=dependency.option_value,
            typeA=dependency.option_type,
            fileA=dependency.option_file,
            technologyA=dependency.option_technology,
            nameB=dependency.dependent_option_name,
            valueB=dependency.dependent_option_value,
            typeB=dependency.dependent_option_type,
            fileB=dependency.dependent_option_file,
            technologyB=dependency.dependent_option_technology,
            dependency_category=dependency.dependency_category
        )