
from plugins.cfgnet_prompt_engine import CfgNetPromptEngine

class PromptEngineFactory:
    def get_prompt_engine(self, tool_name: str):
        if tool_name.lower() == "cfgnet":
            return CfgNetPromptEngine(tool_name=tool_name)
        
        return None
