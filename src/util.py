from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModelType
from llama_index.embeddings.ollama import OllamaEmbedding 
from llama_index.embeddings.huggingface import HuggingFaceEmbedding 
from sentence_transformers import SentenceTransformer
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from typing import Dict
import toml
import os


DIMENSION = {
    "openai": 1536,
    "qwen": 3584,
    "sfr": 4096
}

def get_embedding_dimension(embed_model_name: str) -> int:
    dimension = DIMENSION[embed_model_name]

    if not isinstance(dimension, int):
        raise Exception(f"Dimension has to be an integer and not of type {type(dimension)}")

    return dimension



def load_config(config_file: str) -> Dict:
    """
    Load config from TOML file.
    """
    if not config_file.endswith(".toml"):
            raise Exception("Config file has to be a TOML file.")
        
    with open(config_file, "r", encoding="utf-8") as f:
        config = toml.load(f)
        
    return config


def set_embedding(embed_model_name: str) -> None:
    """
    Set embedding model.
    """
    if embed_model_name == "openai":
        Settings.embed_model = OpenAIEmbedding(
            api_key=os.getenv(key="OPENAI_KEY"),
            model=OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002
        )

    if embed_model_name == "ollama":
        Settings.embed_model = OllamaEmbedding(model_name=embed_model_name)

    if embed_model_name == "qwen":
        Settings.embed_model = HuggingFaceEmbedding(
            "Alibaba-NLP/gte-Qwen2-7B-instruct", 
            trust_remote_code=True
        )
    
    if not Settings.embed_model:
        raise Exception("Embedding model has to be set.")
    


def set_llm(inference_model_name: str) -> None: 
    """
    Set inference model.
    """
    if inference_model_name.startswith("gpt"):
        Settings.llm = OpenAI(
            model=inference_model_name, 
            api_key=os.getenv(key="OPENAI_KEY")
        )
        
    if inference_model_name.startswith("llama"):
        Settings.llm = Ollama(
            model=inference_model_name,
            request_timeout=90.0
    )

    if not Settings.llm:
        raise Exception("Inference model has to be set.")
