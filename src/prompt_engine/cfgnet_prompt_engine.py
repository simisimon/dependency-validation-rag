from llama_index.core import PromptTemplate
from data import Dependency
from plugins.prompt_engine import PromptEngine

class CfgNetPromptEngine(PromptEngine):
    
    value_equality_definition: PromptTemplate = PromptTemplate(
        "A value-equality dependency is present if two configuration options must have identical values in order to function correctly."
        "Inconsistencies in these configuration values can lead to configuration errors."
        "Importantly, configuration options may have equal values by accident, meaning that there is no actual dependency, but it just happens that they have equal values."
    )

    def __init__(self, tool_name: str) -> None:
        super().__init__(tool_name)


    def get_system_str(self, dependency: Dependency) -> str:
        return self.system_prompt.format(
            project=dependency.project,
            dependency_str=self.get_dependency_definition_str(
                dependency_category=dependency.dependency_category
            )
        )
    

    def get_task_str(self, dependency: Dependency) -> str:
        return self.task_prompt.format(
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
    
    def get_dependency_definition_str(self, dependency_category: str) -> str:
        return self.value_equality_definition.format()