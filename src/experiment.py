from typing import Dict, List
from util import load_config
from ingestion_engine import IngestionEngine
from retrieval_engine import RetrievalEngine
from prompt_settings import PrompSettingsFactory
from generator import GeneratorFactory
from pinecone import Pinecone
from dotenv import load_dotenv
from data import Dependency, Response
from cval import CVal
from rich.logging import RichHandler
import mlflow
import pandas as pd
import logging
import json
import glob
import os


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()],
)


def generate(model_name: str, temperature: float, messages: List) -> str:
    """ 
    Generate answer.
    """
    generator = GeneratorFactory().get_generator(
        model_name=model_name,
        temperature=temperature
    )
    logging.info(f"Query {generator.model_name}")
    response = generator.generate(messages=messages)
    return response


def run_experiment(
    config: Dict, 
    index_name: str, 
    model_names: List,
    eval_file_path: str
) -> None:
    """
    Run an experiment.
    """
    file_name = eval_file_path.split("/")[-1].split(".")[0]
    df = pd.read_csv(eval_file_path)
    print("File: ", file_name)

    if os.path.exists(f"../data/evaluation/results/{file_name}_{index_name}.json"):
        print(f"{file_name}_{index_name}.json already exists. Skip file.")
        return

    with mlflow.start_run(run_name=f"{file_name}_{index_name}"): 

        mlflow.log_params(config)

        pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        prompt_settings = PrompSettingsFactory.get_prompt_settings(
            tool_name=config["tool_name"]
        )

        retrieval_engine = RetrievalEngine(
            pinecone_client=pinecone_client,
            with_rewriting=config["with_rewriting"],
            rerank=config["rerank"],
            top_k=config["top_k"],
            top_n=config["top_n"],
            alpha=config["alpha"]
        )

        results = []

        for index, x in enumerate(df.to_dict("records")):
            print("Dependency count: ", index)
            dependency = Dependency(
                project=x["project"],
                option_name=x["option_name"],
                option_value=x["option_value"],
                option_type=x["option_type"].split(".")[-1],
                option_file=x["option_file"],
                option_technology=x["option_technology"],
                dependent_option_name=x["dependent_option_name"],
                dependent_option_value=x["dependent_option_value"],
                dependent_option_type=x["dependent_option_type"].split(".")[-1],
                dependent_option_file=x["dependent_option_file"],
                dependent_option_technology=x["dependent_option_technology"]
            )

            system_str = prompt_settings.get_system_str(dependency=dependency)
            task_str = prompt_settings.get_task_str(dependency=dependency)

            if index == "web-search":
                # TODO (scrape)
                pass

            if config["with_rewriting"]:
                logging.info("Rewrite query.")
                retrieval_str = prompt_settings.get_retrieval_prompt(
                    dependency=dependency
                    )
            else:
                retrieval_str = task_str

            retrieved_nodes = retrieval_engine.retrieve(
                index_name=index_name,
                query_str=retrieval_str
            )

            context_str = "\n\n".join(
                [source_node.node.get_content() for source_node in retrieved_nodes]
            )

            query_str = prompt_settings.query_prompt.format(
                context_str=context_str, 
                task_str=task_str,
                format_str=prompt_settings.get_format_prompt()
            ) 

            messages = [
                {
                    "role": "system", 
                    "content": system_str
                },
                {
                    "role": "user",
                    "content": query_str
                }
            ]

            run_responses = {}
            for model_name in model_names:
                response = Response(
                    input=f"{task_str}\n\n{prompt_settings.get_format_prompt()}",
                    input_complete=query_str,
                    response=generate(
                            model_name=model_name, 
                            temperature=config["temperature"],
                            messages=messages
                        ),
                    source_nodes=retrieved_nodes
                )
                
                run_responses[model_name] = response.to_dict()

            results.append(run_responses)


        with open(f"../data/evaluation/{file_name}_{index_name}.json", "w", encoding="utf-8") as dest:
            json.dump(results, dest, indent=2)

        mlflow.log_artifact(local_path=f"../data/evaluation/{file_name}_{index_name}.json")
        
        print("Done with: ", eval_file_path)


def main():
    config_file = "../config.toml"
    env_file = "../.env"
    eval_data_dir = "../data/evaluation/data"
    index_name = "all"
    eval_file_path = "../data/evaluation/test_dependencies.csv"
    model_names = ["gpt-3.5-turbo-0125", "gpt-4o-2024-05-13", "llama3:70b", "llama3:70b"]


    config = load_config(config_file=config_file)
    load_dotenv(dotenv_path=env_file)

    mlflow.set_experiment(experiment_name=f"inference_{index_name}")

    #for file_path in glob.glob(eval_data_dir + "/**"):
    #    run_experiment(file_path=file_path)

    run_experiment(
        config=config, 
        index_name=index_name,
        model_names=model_names,
        eval_file_path=eval_file_path
    )


if __name__ == "__main__":
    main()



