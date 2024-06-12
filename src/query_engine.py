from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.base.llms.base import BaseLLM
from data import Dependency
from prompt_templates import QUERY_PROMPT, SYSTEM_PROMPT, TASK_PROMPT, VALUE_EQUALITY_DEFINITION_STR, FORMAT_STR
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
        nodes = retriever.retrieve(self.get_task_prompt(dependency=dependency))
        context_str = "\n\n".join([n.node.get_content() for n in nodes])
        query_prompt = QUERY_PROMPT.format(
            system_str=self.get_system_prompt(dependency=dependency),
            context_str=context_str, 
            task_str=self.get_task_prompt(dependency=dependency),
            format_str=FORMAT_STR
        )

        response = llm.complete(
            prompt=query_prompt,
            temperature=temperature
        )
        return response

    def get_system_prompt(self, dependency: Dependency) -> str:
        return SYSTEM_PROMPT.format(
            project=dependency.project,
            definition_str=VALUE_EQUALITY_DEFINITION_STR
        )
    
    def get_task_prompt(self, dependency: Dependency) -> str:
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