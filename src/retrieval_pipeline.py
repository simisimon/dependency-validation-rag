from util import load_config, set_embedding, set_llm
from ingestion import IngestionEngine
from retrieval import RetrievalEngine
from prompt_settings import PrompSettingsFactory
from pinecone import Pinecone
from dotenv import load_dotenv
from typing import Dict
from data import Dependency
import pandas as pd
import os
import toml
import argparse
import json
import mlflow
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="../retrieval_config.toml")
    parser.add_argument("--data_file", type=str, default="../data/evaluation/all_dependencies.csv")   
    
    return parser.parse_args()


def transform(row: pd.Series) -> Dependency:
    dependency = Dependency(
        project=row["project"],
        option_name=row["option_name"],
        option_value=row["option_value"],
        option_type=row["option_type"].split(".")[-1],
        option_file=row["option_file"],
        option_technology=row["option_technology"],
        dependent_option_name=row["dependent_option_name"],
        dependent_option_value=row["dependent_option_value"],
        dependent_option_type=row["dependent_option_type"].split(".")[-1],
        dependent_option_file=row["dependent_option_file"],
        dependent_option_technology=row["dependent_option_technology"]
    )

    return dependency


def run_retrieval(config: Dict, index_name: str, data_file: str):
    
    # set up embedding, llm, and pinecone client
    pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    set_llm(inference_model_name=None)
    set_embedding(embed_model_name=config["embedding_model"])
    dimension = config["embedding_dimension"]

    # set up ingestion engine
    ingestion_engine = IngestionEngine(
        pinecone_client=pinecone_client,
        dimension=dimension,
        splitting=config["splitting"],
        extractors=config["extractors"]
    )

    # set up retrieval engine
    retrieval_engine = RetrievalEngine(
        pinecone_client=pinecone_client,
        rerank=config["rerank"],
        top_k=config["top_k"],
        top_n=config["top_n"],
        alpha=config["alpha"]
    )

    # set up prompt settings
    prompt_settings = PrompSettingsFactory.get_prompt_settings(tool_name=config["tool_name"])
    df = pd.read_csv(data_file)

    queries = []
    counter = 0
    batch_size = 100
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        dependency = transform(row=row)

        system_str = prompt_settings.get_system_str(dependency=dependency)
        task_str = prompt_settings.get_task_str(dependency=dependency)
        retrieval_str = prompt_settings.get_retrieval_prompt(dependency=dependency)

        # scrape web
        if index_name == "web-search" or index_name == "all":
            docs = ingestion_engine.docs_from_web(
                query_str=prompt_settings.get_retrieval_prompt(dependency=dependency), 
                num_websites=config["num_websites"]
            )
            
            for d in docs:
                d.metadata["index_name"] = "web-search"

            ingestion_engine.index_documents(
                index_name="web-search",
                documents=docs,
                delete_index=True
            )

        retrieved_nodes = retrieval_engine.retrieve(
            index_name=index_name,
            query_str=retrieval_str
        )

        context_str = "\n\n".join([source_node.node.get_content() for source_node in retrieved_nodes])

        context = [
            {
                "content": node.get_content(),
                "score": node.get_score(),
                "index": node.metadata["index_name"] if "index_name" in node.metadata else None,
                "id": node.node_id
            } for node in retrieved_nodes
        ]

        queries.append({
            "index": index,
            "dependency": dependency.to_dict(),
            "system_str": system_str,
            "task_str": task_str,
            "context_str": context_str,
            "context": context

        })

        counter += 1

        if index_name == all:
            if counter % batch_size == 0:
                output_file = f"../data/evaluation/all_dependencies_{index_name}_{counter}.json"
                with open(output_file, "a", encoding="utf-8") as dest:
                    json.dump(queries, dest, indent=2)
                    dest.write("\n")  # To separate different batches in the file
                mlflow.log_artifact(local_path=output_file)
                queries.clear()  # Clear the list after writing


    if queries:
        output_file = f"../data/evaluation/all_dependencies_{index_name}.json"
        with open(output_file, "a", encoding="utf-8") as dest:
            json.dump(queries, dest, indent=2)
            dest.write("\n")  # To separate different batches in the file

    mlflow.log_artifact(local_path=output_file)

    print(f"Done with index: {index_name}")


if __name__ == "__main__":
    args = get_args()

    # load config
    config = load_config(config_file=args.config_file)

    mlflow.set_experiment(experiment_name=f"retrieval_{config['rerank']}")

    os.environ["PINECONE_API_KEY"] = config["pinecone_key"]
    print("Pinecone Key: ", os.getenv("PINECONE_API_KEY"))
    
    for index_name in config["index_names"]:

        with mlflow.start_run(run_name=f"{index_name}"): 
            mlflow.log_artifact(local_path=args.config_file)
            mlflow.log_artifact(local_path=args.data_file)
            mlflow.log_params(config)

            run_retrieval(
                config=config, 
                index_name=index_name,
                data_file=args.data_file
            )