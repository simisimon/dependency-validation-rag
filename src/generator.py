from openai import OpenAI, RateLimitError, Timeout, APIError, APIConnectionError
from rich.logging import RichHandler
from typing import Tuple, List
from ollama._types import ResponseError, RequestError
import ollama
import backoff
import logging
import os


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()],
)

class GeneratorFactory:
    def get_generator(self, model_name: str, temperature: int):
        if model_name.startswith("gpt"):
            return GPTGenerator(
                model_name=model_name,
                temperature=temperature
            )
        else:
            return OllamaGenerator(
                model_name=model_name,
                temperature=temperature
            )
        

class Generator:
    def __init__(self, model_name: str, temperature: int) -> None:
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, messages: List) -> Tuple:
        pass


class GPTGenerator(Generator):
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
    ),
    max_tries=10
    )
    def generate(self, messages: List) -> Tuple:
        client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
        response = client.chat.completions.create(
            model=self.model_name, 
            messages=messages,        
            temperature=self.temperature,
            response_format={"type": "json_object"},
            max_tokens=1000
        )

        return response.choices[0].message.content, response.usage   


class OllamaGenerator(Generator):
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
        return response['message']['content'], None
