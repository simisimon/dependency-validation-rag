
from llama_index.core import Settings
from ingestion_engine import IngestionEngine
from dotenv import load_dotenv
from pinecone import Pinecone
from util import set_embedding
from rich.logging import RichHandler
import argparse
import toml
import os
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()],
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="../config.toml")
    parser.add_argument("--env_file", type=str, default="../.env")    
    
    return parser.parse_args()


def run_ingestion(config):
    """Run ingestion pipeline"""

    dimension = set_embedding(embed_model_name=config["general"]["embed_model"])

    pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    ingestion_engine = IngestionEngine(
        pinecone_client=pinecone_client,
        embed_model=Settings.embed_model,
        dimension=dimension,
        splitting=config["ingestion"]["splitting"],
        chunk_size=config["ingestion"]["chunk_size"],
        chunk_overlap=config["ingestion"]["chunk_overlap"],
        extractors=config["ingestion"]["extractors"]
    )

    if pinecone_client.list_indexes().names():
        logging.info("Vector Database is not empty.")

    logging.info("Index data into 'tech-docs'.")
    docs = ingestion_engine.docs_from_urls(urls=config["ingestion"]["urls"])
    ingestion_engine.index_documents(
        index_name="tech-docs",
        documents=docs,
        delete_index=True
    )

    logging.info("Index data into 'so-posts'.")
    docs = ingestion_engine.docs_from_dir(data_dir=config["ingestion"]["data_dir"])
    ingestion_engine.index_documents(
        index_name="so-posts",
        documents=docs,
            delete_index=True
    )

    logging.info("Index data into 'github'.")
    docs = []
    for project_name in config["ingestion"]["github"]:
        docs += ingestion_engine.docs_from_github(project_name=project_name)

    ingestion_engine.index_documents(
        index_name="github",
        documents=docs,
        delete_index=True
    )


if __name__ == "__main__":
    args = get_args()

    # load env variables
    load_dotenv(dotenv_path=args.env_file)

    # load config
    with open(args.config_file, "r", encoding="utf-8") as f:
        config = toml.load(f)

    run_ingestion(config=config)
