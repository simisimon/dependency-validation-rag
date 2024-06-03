from dotenv import dotenv_values, load_dotenv
from database.database import VectorDatabase
import argparse
import os


def get_args():
    parser = argparse.ArgumentParser()
    # basic RAG config
    parser.add_argument("--enable_rag", type=bool, default=True)
    parser.add_argument("--embed_model", type=str, default="openai", choices=["openai", "huggingface"])
    parser.add_argument("--mode", type=str, default="static", choices=["static", "live"])

    # vector database config
    parser.add_argument("--index_name", type=str, default="technology-docs", choices=["technology-docs", "so-posts", "blog-posts", "websearch"])
    parser.add_argument("--metric", type=str, default="cosine")
    parser.add_argument("--dimension", type=int, default=1536)
    
    # retrieval config
    parser.add_argument("--top_k", type=int, default=5)

    # llm config
    parser.add_argument("--model_name", type=str, default="llama3:70b", choices=["gpt-4-0125-preview", "gpt-3.5-turbo-0125", "llama3:8b", "gpt-4o-2024-05-13"])
    parser.add_argument("--prompt_strategy", type=str, default="zero-shot", choices=[])
    parser.add_argument("--temperature", type=float, default=0.0, choices=[0.0, 0.2, 0.7, 1.0])
   
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


    #print(os.getenv("PINECONE_API_KEY"))
    #print(config["OPENAI_KEY"])


    # TODO: get dependencies from software project
    # TODO: select sample set
    # TODO: validate sample set
    ## TODO: get context information
    ## TODO: create prompt
    ## TODO: utilize LLMS
    ## TODO: store results


if __name__ == "__main__":
    main()