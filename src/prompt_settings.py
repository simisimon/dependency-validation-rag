from llama_index.core import PromptTemplate
from typing import Optional
from data import Dependency
from dataclasses import dataclass
from rich.logging import RichHandler
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()],
)


class _PromptSettings:
    _query_prompt: PromptTemplate = PromptTemplate(
        "Information about both configuration options, such as their descriptions or prior usages are below:\n\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information, perform the following task:\n\n"
        "{task_str}\n\n"
        "{format_str}"
    )

    _advanced_query_prompt: PromptTemplate = PromptTemplate(
        "Information about both configuration options, including their descriptions, prior usages, and examples of similar dependencies are provided below.\n"
        "The provided information comes from various sources, such as manuals, Stack Overflow posts, GitHub repositories, and web search results.\n"
        "Note that not all the provided information may be relevant for validating the dependency.\n"
        "Consider only the information that is relevant for validating the dependency, and disregard the rest."
        "{context_str}\n"
        "Additionally, here are some examples on how similar dependencies are evaluated:\n\n"
        "{shot_str}\n"
        "---------------------\n"
        "Given the information and examples, perform the following task:\n\n"
        "{task_str}\n\n"
        "{format_str}"
    )

    _format_str: Optional[PromptTemplate] =  None
    _system_prompt: Optional[PromptTemplate] = None
    _task_prompt: Optional[PromptTemplate] = None
    _dependency_prompt: Optional[PromptTemplate] = None 
    _retrieval_prompt: Optional[PromptTemplate] = None

    @property
    def query_prompt(self) -> Optional[PromptTemplate]:
        """Get the query prompt."""
        return self._query_prompt

    @property
    def advanced_query_prompt(self) -> Optional[PromptTemplate]:
        """Get the query prompt."""
        return self._advanced_query_prompt

    @property
    def system_prompt(self) -> Optional[PromptTemplate]:
        """Get the system prompt."""
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, system_prompt: PromptTemplate) -> None:
        """Set the system prompt."""
        self._system_prompt = system_prompt


    @property
    def task_prompt(self) -> Optional[PromptTemplate]:
        """Get the task prompt."""
        return self._task_prompt

    @task_prompt.setter
    def task_prompt(self, task_prompt: PromptTemplate) -> None:
        """Set the task prompt."""
        self._task_prompt = task_prompt


    @property
    def dependency_prompt(self) -> Optional[PromptTemplate]:
        """Get the dependency prompt."""
        return self._task_prompt

    @dependency_prompt.setter
    def dependency_prompt(self, dependency_prompt: PromptTemplate) -> None:
        """Set the task prompt."""
        self._dependency_prompt = dependency_prompt

    @property
    def format_str(self) -> Optional[PromptTemplate]:
        """Get the format prompt"""
        return self._format_str
    
    @format_str.setter
    def format_prompt(self, format_prompt: PromptTemplate) -> None:
        """Set the format prompt."""
        self._format_str = format_prompt
    
    @property
    def retrieval_prompt(self) -> Optional[PromptTemplate]:
        """Get the dependency prompt."""
        return self._retrieval_prompt
    
    @retrieval_prompt.setter
    def retrieval_prompt(self, retrieval_prompt: PromptTemplate) -> None:
        """Set the task prompt."""
        self._retrieval_prompt = retrieval_prompt


    def get_system_str(self, dependency: Dependency) -> str:
        """Get formatted system prompt."""
        pass
            
    def get_task_str(self, dependency: Dependency) -> str:
        """Get formatted system task prompt."""
        pass
        
    def get_dependency_definition_str(self, dependency: Dependency) -> str:
        """Get formatted dependency prompt."""
        pass

    def get_retrieval_prompt(self, dependency: Dependency) -> str:
        """Get formatted dependency prompt."""
        pass

    def get_format_prompt(self) -> str:
        return self._format_str.format()



class _AdvancedCfgNetPromptSettings(_PromptSettings):
    _system_prompt: Optional[PromptTemplate] = PromptTemplate(
        "You are a full-stack expert in validating intra-technology and cross-technology configuration dependencies.\n" 
        "You will be presented with configuration options found in the software project '{project}'.\n\n" 
        "{project_str}\n\n"
        "Your task is to determine whether the given configuration options actually depend on each other based on value-equality.\n"
        "{dependency_str}"
    )

    _task_prompt: Optional[PromptTemplate] = PromptTemplate(
        "Carefully evaluate whether configuration option {nameA} of type {typeA} with value {valueA} in {fileA} of technology {technologyA} "
        "depends on configuration option {nameB} of type {typeB} with value {valueB} in {fileB} of technology {technologyB} or vice versa." 
    )

    _dependency_prompt: Optional[PromptTemplate] = PromptTemplate(
        "A value-equality dependency is present if two configuration options must have identical values in order to function correctly.\n"
        "Inconsistencies in these configuration values can lead to configuration errors.\n"
        "Importantly, configuration options may have equal values by accident, meaning that there is no actual dependency, but it just happens that they have equal values.\n"
        "If the values of configuration options are identical merely to ensure consistency within a software project, the options are not considered dependent."
    )

    _retrieval_prompt: Optional[PromptTemplate] = PromptTemplate(
        "Dependency between {nameA} in {technologyA} with value {valueA} and {nameB} in {technologyB} with value {valueB}"
    )

    _format_str: Optional[PromptTemplate] = PromptTemplate(
        "Respond in a JSON format as shown below:\n"
        "{{\n"
        "\t“plan”: string, // Write down a step-by-step plan on how to solve the task given the information and examples of similar dependencies above.\n"
        "\t“rationale”: string, // Provide a concise explanation of whether and why the configuration options depend on each other due to value-equality.\n"
        "\t“isDependency”: boolean // True if a dependency exists, or False otherwise.\n"
        "}}"
    )




    def get_system_str(self, dependency: Dependency, project_str: str) -> str:
        """Get formatted system prompt."""
        return self._system_prompt.format(
            project=dependency.project,
            project_str=project_str,
            dependency_str=self.get_dependency_definition_str(
                dependency=dependency
            )
        )
        
    def get_task_str(self, dependency: Dependency) -> str:
        """Get formatted system task prompt."""
        return self._task_prompt.format(
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
        
    def get_dependency_definition_str(self, dependency: Dependency) -> str:
        """Get formatted dependency prompt."""
        return self._dependency_prompt.format()
    
    def get_retrieval_prompt(self, dependency: Dependency) -> str:
        """Get formatted retrieval prompt."""
        return self._retrieval_prompt.format(
            nameA=dependency.option_name,
            technologyA=dependency.option_technology,
            valueA=dependency.option_value,
            nameB=dependency.dependent_option_name,
            technologyB=dependency.dependent_option_technology,
            valueB=dependency.dependent_option_value
        )



class _CfgNetPromptSettings(_PromptSettings):

    _advanced_query_prompt: PromptTemplate = PromptTemplate(
        "Information about both configuration options, including their descriptions, prior usages, and examples of similar dependencies are provided below.\n"
        "This information comes from various sources, such as manuals, Stack Overflow posts, GitHub repositories, and web search results.\n"
        "Be aware that not all the provided information may be relevant for validating the dependency.\n"
        "Consider only the information that is relevant for validating the dependency, and disregard the rest."
        "{context_str}\n"
        "Additionally, here are some examples on how similar dependencies are assessed.\n"
        "{shot_str}"
        "---------------------\n"
        "Given the information and examples, perform the following task:\n\n"
        "{task_str}\n\n"
        "{format_str}"
    )

    _advanced_system_prompt: Optional[PromptTemplate] = PromptTemplate(
        "You are a full-stack expert in validating intra-technology and cross-technology configuration dependencies.\n" 
        "You will be presented with configuration options found in the software project '{project}'.\n\n" 
        "{project_str}\n\n"
        "Your task is to determine whether the given configuration options actually depend on each other based on value-equality.\n\n"
        
        "{dependency_str}"
    )


    _system_prompt: Optional[PromptTemplate] = PromptTemplate(
        "You are a full-stack expert in validating intra-technology and cross-technology configuration dependencies.\n" 
        "You will be presented with configuration options found in the software project '{project}'.\n" 
        "Your task is to determine whether the given configuration options actually depend on each other based on value-equality.\n\n"
        "{dependency_str}"
    )

    _task_prompt: Optional[PromptTemplate] = PromptTemplate(
        "Carefully evaluate whether configuration option {nameA} of type {typeA} with value {valueA} in {fileA} of technology {technologyA} "
        "depends on configuration option {nameB} of type {typeB} with value {valueB} in {fileB} of technology {technologyB} or vice versa." 
    )

    _dependency_prompt: Optional[PromptTemplate] = PromptTemplate(
        "A value-equality dependency is present if two configuration options must have identical values in order to function correctly.\n"
        "Inconsistencies in these configuration values can lead to configuration errors.\n"
        "Importantly, configuration options may have equal values by accident, meaning that there is no actual dependency, but it just happens that they have equal values."
    )

    _advanced_dependency_prompt: Optional[PromptTemplate] = PromptTemplate(
        "A value-equality dependency is present if two configuration options must have identical values in order to function correctly.\n"
        "Inconsistencies in these configuration values can lead to configuration errors.\n"
        "Importantly, configuration options may have equal values by accident, meaning that there is no actual dependency, but it just happens that they have equal values.\n"
        "If the values of configuration options are identical merely to ensure consistency within a software project, the options are not considered dependent."
    )

    _retrieval_prompt: Optional[PromptTemplate] = PromptTemplate(
        "Dependency between {nameA} in {technologyA} with value {valueA} and {nameB} in {technologyB} with value {valueB}"
    )

    _format_str: Optional[PromptTemplate] = PromptTemplate(
        "Respond in a JSON format as shown below:\n"
        "{{\n"
        "\t“plan”: string, // Write down a step-by-step plan on how to solve the task given the information above.\n"
        "\t“rationale”: string, // Provide a concise explanation of whether and why the configuration options depend on each other due to value-equality.\n"
        "\t“uncertainty”: integer, // Rate your certainty of this dependency on a scale from 0 (completely uncertain) to 10 (absolutely certain), given the context, plan, and rationale.\n"
        "\t“isDependency”: boolean // True if a dependency exists, or False otherwise.\n"
        "}}"
    )

    _advanced_format_str: Optional[PromptTemplate] = PromptTemplate(
        "Respond in a JSON format as shown below:\n"
        "{{\n"
        "\t“plan”: string, // Write down a step-by-step plan on how to solve the task given the information above.\n"
        "\t“rationale”: string, // Provide a concise explanation of whether and why the configuration options depend on each other due to value-equality.\n"
        "\t“isDependency”: boolean // True if a dependency exists, or False otherwise.\n"
        "}}"
    )

    def get_advanced_system_str(self, dependency: Dependency, ) -> str:
        """Get formatted system prompt."""
        return self._adsystem_prompt.format(
            project=dependency.project,
            dependency_str=self.get_dependency_definition_str(
                dependency=dependency
            )
        )

    def get_system_str(self, dependency: Dependency) -> str:
        """Get formatted system prompt."""
        return self._system_prompt.format(
            project=dependency.project,
            dependency_str=self.get_dependency_definition_str(
                dependency=dependency
            )
        )
        
    def get_task_str(self, dependency: Dependency) -> str:
        """Get formatted system task prompt."""
        return self._task_prompt.format(
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
        
    def get_dependency_definition_str(self, dependency: Dependency) -> str:
        """Get formatted dependency prompt."""
        return self._dependency_prompt.format()
    
    def get_retrieval_prompt(self, dependency: Dependency) -> str:
        """Get formatted retrieval prompt."""
        return self._retrieval_prompt.format(
            nameA=dependency.option_name,
            technologyA=dependency.option_technology,
            valueA=dependency.option_value,
            nameB=dependency.dependent_option_name,
            technologyB=dependency.dependent_option_technology,
            valueB=dependency.dependent_option_value
        )
    

# Singelton
CfgNetPromptSettings = _CfgNetPromptSettings()
AdvancedCfgNetPromptSettings = _AdvancedCfgNetPromptSettings()
    

class PrompSettingsFactory:
    @staticmethod
    def get_prompt_settings(tool_name: str):
        if tool_name.lower() == "cfgnet":
            logging.info("Initialize CfgNet prompt settings.")
            return CfgNetPromptSettings
        else:
            return None


