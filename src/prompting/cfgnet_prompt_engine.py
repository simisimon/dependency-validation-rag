from llama_index.core import PromptTemplate
from prompt_engine import PromptEngine

class CfgNetPromptEngine(PromptEngine):
    system_prompt = PromptTemplate(
        "You are a full-stack expert in validating intra-technology and cross-technology configuration dependencies.\n" 
        "You will be presented with configuration options found in the software project '{project}'.\n" 
        "Your task is to determine whether the given configuration options actually depend on each other based on value-equality.\n\n"
        "{dependency_str}"
    )

    task_prompt = PromptTemplate(
        "Carefully evaluate whether configuration option {nameA} of type {typeA} with value {valueA} in {fileA} of technology {technologyA} \
        depends on configuration option {nameB} of type {typeB} with value {valueB} in {fileB} of technology {technologyB} or vice versa." 
    )


    def __init__(self, name: str) -> None:
        super().__init__(name)


    