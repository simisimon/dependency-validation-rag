from deepeval.models.base_model import DeepEvalBaseLLM
from openai import OpenAI

class OpenAIEvalModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model
    ):
        self.model = model

    def load_model(self):
        return self.model
    
    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = chat_model.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="gpt-4o-2024-05-13",
            temperature=0.0
        )

        return res.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "OpenAI Eval Model"
    


class LlamaEvalModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model
    ):
        self.model = model

    def load_model(self):
        return self.model
    
    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = chat_model.chat(
            model="llama3:70b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0.0
            }

        )

        return res['message']['content']

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Llama3 Eval Model"