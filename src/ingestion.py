from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.readers.web import SimpleWebPageReader
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.core.extractors import TitleExtractor
from splitting import DataSplittingFactory
from data import Dependency
from prompt_templates import SCRAPING_PROMPT
from typing import List
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
from rich.logging import RichHandler
import glob
import re
import backoff
import requests
import logging
import os
import json


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()],
)


class DataIngestionEngine:
    def __init__(self) -> None:
        logging.info(f"Data Ingestion Engine initialized.")
    
    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=8,
    )
    def docs_from_web(
        self, 
        dependency: Dependency,
        num_websites: int
    ) -> List:
        """
        Get documents from websites.
        """
        logging.info(f"Start scraping {num_websites} websites.")
        scraping_query = SCRAPING_PROMPT.format(
            technologyA=dependency.option_technology,
            nameA=dependency.option_name,
            technologyB=dependency.dependent_option_technology,
            nameB=dependency.dependent_option_name
        )

        url = "https://www.bing.com/search?q=" + scraping_query

        response = requests.get(
            url, 
            headers={'User-Agent':  UserAgent().chrome}
        )
        response.raise_for_status()
        
        search_urls = []

        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all("h2")

        links = [x for x in links if "https://" in str(x)]

        for link in links:
            match = re.findall(r'href="([^"]*)"', str(link))
            search_urls.append(match[0])

        if not search_urls or len(search_urls) < num_websites:
            raise Exception() 
        
        documents = SimpleWebPageReader(html_to_text=True).load_data(search_urls)

        return documents
    
    def docs_from_github(self, dependency: Dependency) -> List:
        """
        Get documents from GitHub repository.
        """
        logging.info(f"Start scraping the repository of {dependency.project}.")
        response = requests.get(f"https://api.github.com/search/repositories?q={dependency.project}")
        response.raise_for_status()

        data = response.json()

        if data['total_count'] > 0:
            owner = data['items'][0]['owner']['login']
            branch = data['items'][0]['default_branch']
            topics = data['items'][0]["topics"]
            full_name = data['items'][0]["full_name"]
            description = data['items'][0]["description"]
        else:
            return []
        
        github_client = GithubClient(github_token=os.getenv(key="GITHUB_TOKEN"), verbose=True)
        
        try:
            documents = GithubRepositoryReader(
                github_client=github_client,
                owner=owner,
                repo=dependency.project,
                use_parser=False,
                verbose=False,
                filter_file_extensions=(
                    [
                        ".xml",
                        ".properties",
                        ".yml",
                        "Dockerfile",
                        ".json",
                        ".ini",
                        ".cnf",
                        ".toml",
                        ".yaml",
                        ".conf",
                        ".md"
                    ],
                    GithubRepositoryReader.FilterType.INCLUDE,
                ),
            ).load_data(branch=branch)

            for d in documents:
                d.metadata["description"] = description
                d.metadata["full_name"] = full_name
                d.metadata["topics"] = topics
            
            return documents
        except Exception:
            logging.info("Error occurred while scraping Github.")
            return []

    def docs_from_dir(self, directory: str) -> List:
        """
        Create documents from directory.
        """
        if not os.path.exists(directory):
            raise Exception(f"Dir {directory} does not exist.")
            
        documents = SimpleDirectoryReader(input_dir=directory, recursive=True).load_data()

        return documents
        
    def docs_from_urls(self, url_file: str) -> List:
        """
        Get documents from urls.
        """
        if not url_file.endswith(".json"):
            raise Exception(f"URL file has be a JSON file.")
        
        with open(url_file, "r", encoding="utf-8") as src:
            data = json.load(src)

        urls = data["urls"]
        documents = SimpleWebPageReader(html_to_text=True).load_data(urls)
        for document in documents: 
            document.metadata["file_name"] = document.id_

        return documents

    def index_documents(
            self,
            vector_store,
            documents: List,
            splitting: str = "sentence",
    ) -> None:
        """
        Add documents to a vector store
        """
        transformations = [
            DataSplittingFactory().get_splitting(splitting=splitting),
            Settings.embed_model,
        ]

        pipeline = IngestionPipeline(
            transformations=transformations,
            vector_store=vector_store
        )
        pipeline.run(documents=documents)
    


    
      
