from dotenv import dotenv_values, load_dotenv
from database.database import VectorDatabase
from typing import Dict
import argparse
import os


class CVal:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self._check_config()

    def _check_config(self):
        raise NotImplementedError()

    def validate():
        raise NotImplementedError


def get_args():
    parser = argparse.ArgumentParser()

    # basic CVal config
    parser.add_argument(
        "--output_file", 
        type="str",
        help="Path to the output file or directory."
    )

    # basic RAG config
    parser.add_argument(
        "--enable_rag", 
        type=bool, 
        default=True,
        help="Enable or disable the RAG system."
    )
    parser.add_argument(
        "--embed_model", 
        type=str, 
        default="openai", 
        choices=["openai", "huggingface"],
        help="Embedding model used for vectorization."
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        default="static", 
        choices=["static", "live"],
        help="Operation mode of CVal."
    )

    # vector database config
    parser.add_argument(
        "--index_name", 
        type=str, 
        default="technology-docs", 
        choices=["technology-docs", "so-posts", "blog-posts", "websearch"],
        help="Name of the index."
    )
    parser.add_argument(
        "--metric", 
        type=str, 
        default="cosine",
        help="Metric of the index."
    )
    parser.add_argument(
        "--dimension", 
        type=int, 
        default=1536,
        help="Dimension of the index."
    )
    
    # retrieval config
    parser.add_argument(
        "--top_k",
        type=int, 
        default=5,
        help="Number of similar documents to retrieve."
    )

    # llm config
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="llama3:70b", 
        choices=["gpt-4-0125-preview", "gpt-3.5-turbo-0125", "llama3:8b", "gpt-4o-2024-05-13"],
        help="Name of the model"
    )
    parser.add_argument(
        "--prompt_strategy",
        type=str, 
        default="zero-shot",
        choices=[],
        help="Name of the prompting strategy."
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.0,
        help="Temperature of the model."
    )
   
    return parser.parse_args()


def main():
    args = get_args()
    load_dotenv()

    db = VectorDatabase()
    retrieval_engine = db.get_retrieval_engine(args)

    answer = retrieval_engine.retrieve('EXPOSE instruction Docker')

    # Inspect results
    for i in answer:
        print(i.get_content())




if __name__ == "__main__":
    main()