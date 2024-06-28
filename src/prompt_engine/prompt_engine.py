from llama_index.core import PromptTemplate
from data import Dependency


class PromptEngine:
    system_prompt: PromptTemplate = PromptTemplate(
        "You are a full-stack expert in validating intra-technology and cross-technology configuration dependencies.\n" 
        "You will be presented with configuration options found in the software project '{project}'.\n" 
        "Your task is to determine whether the given configuration options actually depend on each other based on value-equality.\n\n"
        "{dependency_str}"
    )

    task_prompt: PromptTemplate = PromptTemplate(
        "Carefully evaluate whether configuration option {nameA} of type {typeA} with value {valueA} in {fileA} of technology {technologyA}"
        "depends on configuration option {nameB} of type {typeB} with value {valueB} in {fileB} of technology {technologyB} or vice versa." 
    )

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name

    def get_system_str(self, dependency: Dependency) -> str:
        """
        Get system message. 
        """
        pass

    def get_task_str(self, dependency: Dependency) -> str:
        """
        Get task message.
        """
        pass

    def get_dependency_definition_str(self, dependency_category: str) -> str:
        """
        Get dependency definition.
        """
        pass
