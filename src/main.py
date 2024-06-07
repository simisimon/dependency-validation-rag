from dotenv import load_dotenv
from data import Dependency, DependencyLevel, DependencyType
from cval import CVal
import argparse
import json


TEST_PROMPT = """Is there a configuration dependency between the port specified in my Dockerfile and in the application.yml of my Webserver?

Respond in a JSON format as shown below:
{{
  "rationale": string, // Provide a concise explanation of whether and why the configuration options depend on each other.
  "uncertainty": integer, // Rate your certainty of this dependency on a scale from 0 (completely uncertain) to 10 (absolutely certain).
  "isDependency": boolean // Indicate True if a dependency exists, or False otherwise.
}}"""

def get_args():
    parser = argparse.ArgumentParser()

    # basic CVal config
    parser.add_argument("--output_file", type=str, help="Path of output file or directory.")

    # scraping config
    parser.add_argument("--num_websites", type=int, default=5, help="Number of websites to be scraped.")

    # vector database config
    parser.add_argument("--index_name", type=str, default="tech-docs",  choices=["tech-docs", "so-posts", "blog-posts", "web-search"], help="Name of the index.")

    # retrieval config
    parser.add_argument("--top_k", type=int, default=10, help="Number of documents to retrieve.")
    parser.add_argument("--retriever_type", type=str, default="rerank_and_filter_retriever", choices=["base_retreiver", "rerank_retriever", "rerank_and_filter_retriever", "auto_merging_retriever"], help="Type of retriever.")

    # llm config
    parser.add_argument("--model_name", type=str, default="gpt-4o-2024-05-13", choices=["gpt-4-0125-preview", "gpt-3.5-turbo-0125", "llama3:8b", "gpt-4o-2024-05-13"], help="Name of the model")
    parser.add_argument("--prompt_strategy", type=str, default="zero-shot", choices=[], help="Name of the prompting strategy.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature of the model.")
   
    return parser.parse_args()


def main():
    load_dotenv()
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

    cval = CVal(
        model_name=args.model_name,
        env_file_path="./.env"
    )

    cval.validate(
        query=TEST_PROMPT,
        index_name=args.index_name,
        retriever_type=args.retriever_type,
        top_k=args.top_k
    )


if __name__ == "__main__":
    main()