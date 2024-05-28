from openai import OpenAI, RateLimitError, Timeout, APIError, APIConnectionError
from rich.logging import RichHandler
from typing import Tuple, Dict, List
from ollama._types import ResponseError, RequestError
import ollama
import backoff
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()],
)

class GeneratorFactory:
    def get_generator(self, model_name):
        if model_name.startswith("gpt"):
            return GPTGenerator(model_name=model_name)
        else:
            return OllamaGenerator(model_name=model_name)
        

class Generator:
    def __init__(self, model_name) -> None:
        self.model_name = model_name

    def generate(self, args: Dict, messages: List) -> Tuple:
        pass


class GPTGenerator(Generator):
    def __init__(self, model_name) -> None:
        super().__init__(model_name)
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
    def generate(self, args: Dict, messages: List) -> Tuple:
        client = OpenAI(api_key=args.api_key)
        response = client.chat.completions.create(
            model=args.model_name, 
            messages=messages,        
            temperature=args.temperature,
            response_format={"type": "json_object"},
            max_tokens=1000
        )

        return response.choices[0].message.content, response.usage   


class OllamaGenerator(Generator):
    def __init__(self, model_name) -> None:
        super().__init__(model_name)
        logging.info(f"Ollama ({model_name}) generator initialized.")

    @backoff.on_exception(
    backoff.expo,
    (
        ResponseError,
        RequestError
    ),
    max_tries=10
    )
    def generate(self, args: Dict, messages: List) -> Tuple:
        response = ollama.chat(
            model=args.model_name, 
            messages=messages,
            format="json",
            options={
                "temperature": args.temperature
            }
        )
        return response['message']['content'], None
