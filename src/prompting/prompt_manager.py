from templates.templates import SYSTEM_PROMPT, USER_PROMPT, DEPENDENCY_PROMPT
from data.dependency import Dependency
from typing import List


class PromptManager:

    @staticmethod
    def create_prompt(dependency: Dependency, context_str: List) -> List:
        """
        Create prompt.
        """
        
        dependency_description = DEPENDENCY_PROMPT.format(
            
        )

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": ""
            }
        ]

        return messages





