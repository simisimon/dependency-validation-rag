from dotenv import load_dotenv
from data.dependency import Dependency, DependencyCategory, DependencyLevel, DependencyType
from cval import CVal
import argparse
import json


def get_args():
    parser = argparse.ArgumentParser()

    # basic CVal config
    parser.add_argument(
        "--output_file", 
        type=str,
        help="Path of output file or directory."
    )

    # basic RAG config
    parser.add_argument(
        "--enable_rag", 
        type=bool, 
        default=True,
        help="Enable or disable the RAG system."
    )
    parser.add_argument(
        "--embed_model_name", 
        type=str, 
        default="openai", 
        choices=["openai", "huggingface"],
        help="Embedding model used for vectorization."
    )

    # scraping config
    parser.add_argument(
        "--num_documents", 
        type=int, 
        default=5,
        help="Number of documented to be scraped."
    )

    # vector database config
    parser.add_argument(
        "--index_name", 
        type=str, 
        default="technology-docs", 
        choices=["technology-docs", "so-posts", "blog-posts", "web-search"],
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

    dep = Dependency(
        dependency_type=DependencyType.INTRA.name,
        dependency_level=DependencyLevel.CONFIG_FILE_LEVEL.name,
        project="piggymetrics",
        dependency_category="value-equality",
        option_name="EXPOSE",
        option_value="8080",
        option_type="PORT",
        option_file="Dockerfile",
        option_technology="Docker",
        dependent_option_name="server.port",
        dependent_option_value="8080",
        dependent_option_file="application.yml",
        dependent_option_type="PORT",
        dependent_option_technology="Spring-Boot"
    )

    cval = CVal(config=args)

    context = cval.retrieve("EXPOSE server.port dependency")

    source_nodes = [i.get_content() for i in context.source_nodes]

    for response, (_, values) in zip(source_nodes, context.metadata.items()):
        print(response, json.dumps(values, indent=2))
        break


if __name__ == "__main__":
    main()