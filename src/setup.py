
from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModelType
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from ingestion import IngestionEngine
from llama_index.core import Settings
from dotenv import load_dotenv
from pinecone import Pinecone
from typing import Dict, Tuple
import argparse
import toml
import torch
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="../config.toml")
    parser.add_argument("--embed_model_name", type=str, default="qwen", choices=["openai", "ollama", "qwen"])
    parser.add_argument("--env_file", type=str, default="../.env")
    parser.add_argument("--splitting", type=str, default="sentence", choices=["token", "sentence", "recursive", "semantic"])
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--chunk_overlap", type=int, default=50)
    parser.add_argument("--extractors", type=list, default=[]) 
    
    
    return parser.parse_args()



def get_embedding_model(embed_model_name: str) -> Tuple:
    """
    Get embedding model and corresponding dimension.
    """
    embed_model = None
    dimension = None

    if embed_model_name == "openai":
        embed_model = OpenAIEmbedding(
            api_key=os.getenv(key="OPENAI_KEY"),
            model=OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002
        )
        dimension = 1536

    if embed_model_name == "qwen":
        embed_model = HuggingFaceEmbedding(
            model_name="Alibaba-NLP/gte-Qwen2-7B-instruct",
            trust_remote_code=True
        )
        dimension = 3584
    
    return embed_model, dimension




def run_ingestion(args):

    #embed_model, dimension = get_embedding_model(embed_model_name=args.embed_model_name)

    pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    print(torch.cuda.is_available())

    return

    ingestion_engine = IngestionEngine(
        pinecone_client=pinecone_client,
        embed_model=embed_model,
        dimension=dimension,
        splitting=args.splitting,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        extractors=args.extractors
    )


if __name__ == "__main__":
    args = get_args()

    # load config
    with open(args.config_file, "r", encoding="utf-8") as f:
        config = toml.load(f)

    # load env variables
    load_dotenv(dotenv_path=args.env_file)

    run_ingestion(args=args)
