from llama_index.core import PromptTemplate
from cfgnet_prompt_engine import CfgNetPromptEngine
from data import Dependency


class PromptEngineFactory:
    def get_prompt_engine(self, tool_name: str):
        if tool_name == "CfgNet":
            return CfgNetPromptEngine(tool_name=tool_name)
        
        return None


class PromptEngine:
    system_prompt: PromptTemplate
    task_prompt: PromptTemplate

    def __init__(self, name: str, category) -> None:
        self.name = name
        self.category = category

    def get_system_str(self, dependency: Dependency) -> str:
        pass

    def get_task_str(self, dependnecy: Dependency) -> str:
        pass
