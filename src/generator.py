from openai import OpenAI, RateLimitError, Timeout, APIError, APIConnectionError
from rich.logging import RichHandler
from typing import Tuple, List, Dict
from ollama._types import ResponseError, RequestError
from anthropic import Anthropic
import ollama
import backoff
import logging
import os
import json


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()],
)

class GeneratorFactory:
    def get_generator(self, model_name: str, temperature: int):
        if model_name.startswith("gpt"):
            return GPTGeneratorEngine(
                model_name=model_name,
                temperature=temperature
            )
        if model_name.startswith("llama"):
            return OllamaGeneratorEngine(
                model_name=model_name,
                temperature=temperature
            )

        if model_name.startswith("claude"):
            return AnthropicGeneratorEngine(
                model_name=model_name,
                temperature=temperature
            )
    
        else:
            raise Exception(f"Model {model_name} is not yet supported.")
        

class GeneratorEngine:
    def __init__(self, model_name: str, temperature: int) -> None:
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, messages: List) -> str:
        pass



class AnthropicGeneratorEngine(GeneratorEngine):
    def __init__(self, model_name: str, temperature: int) -> None:
        super().__init__(
            model_name=model_name, 
            temperature=temperature
        )
        logging.info(f"Anthropic ({model_name}) generator initialized.")

    
    def generate(self, messages: List) -> str:
        client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            max_tokens=1000,
            system=messages[0]["content"],
            messages=[
                {
                    "role": messages[1]["role"],
                    "content": messages[1]["content"],
                }
            ],
            model=self.model_name,
            temperature=self.temperature
        )

        #x = response.content[0].to_dict()
        #print("Response: ", x, type(x))

        return response.content[0].to_dict()["text"]


class GPTGeneratorEngine(GeneratorEngine):
    def __init__(self, model_name: str, temperature: int) -> None:
        super().__init__(
            model_name=model_name, 
            temperature=temperature
        )
        logging.info(f"GPT ({model_name}) generator initialized.")

    @backoff.on_exception(
    backoff.expo,
    (
        RateLimitError,
        APIError,
        APIConnectionError,
        Timeout,
        Exception
    ),
    max_tries=5
    )
    def generate(self, messages: List) -> str:
        client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
        response = client.chat.completions.create(
            model=self.model_name, 
            messages=messages,        
            temperature=self.temperature,
            response_format={"type": "json_object"},
            max_tokens=1000
        )
    
        response_content = response.choices[0].message.content

        if not response or len(response_content.strip()) == 0:
            raise Exception("Response content was empty.")
        
        return response_content


class OllamaGeneratorEngine(GeneratorEngine):
    def __init__(self, model_name: str, temperature: int) -> None:
        super().__init__(
            model_name=model_name, 
            temperature=temperature
        )
        logging.info(f"Ollama ({model_name}) generator initialized.")

    @backoff.on_exception(
    backoff.expo,
    (
        ResponseError,
        RequestError
    ),
    max_tries=10
    )
    def generate(self, messages: List) -> Tuple:
        response = ollama.chat(
            model=self.model_name, 
            messages=messages,
            format="json",
            options={
                "temperature": self.temperature
            }
        )
        return response['message']['content']
