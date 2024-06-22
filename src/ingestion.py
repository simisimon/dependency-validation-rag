from llama_index.core import Settings, SimpleDirectoryReader, Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.readers.web import SimpleWebPageReader
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.core.extractors import TitleExtractor
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter, SemanticSplitterNodeParser, LangchainNodeParser
from llama_index.core.extractors import SummaryExtractor, TitleExtractor, KeywordExtractor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from weaviate import WeaviateClient
from data import Dependency
from prompt_templates import SCRAPING_PROMPT
from typing import List
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
from rich.logging import RichHandler
import weaviate
import re
import backoff
import requests
import logging
import os


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()],
)


class IngestionEngine:
    def __init__(self, weaviate_client: WeaviateClient) -> None:
        self._weaviate_client = weaviate_client
        self._llm = Settings.llm
        self._embed_model = Settings.embed_model
        
    def _connect_to_weaviate_db(self) -> WeaviateClient:
        """
        Create weaviate client.
        """
        weaviate_client = weaviate.connect_to_custom(
            http_host=self.weaviate_host,
            http_port=self.weaviate_http_port,
            http_secure=False,
            grpc_host=self.weaviate_host,
            grpc_port=self.weaviate_grpc_port,
            grpc_secure=False,
        )
        return weaviate_client

    def _create_node_parser(self, splitting: str):
        node_parser = None

        if splitting == "token":
            node_parser = TokenTextSplitter(
                chunk_size=512,
                chunk_overlap=50,
                separator=" ",
            )

        if splitting == "sentence":
            node_parser = SentenceSplitter(
                chunk_size=512, 
                chunk_overlap=50
            )

        if splitting == "semantic":
            node_parser = SemanticSplitterNodeParser(
                buffer_size=1, 
                breakpoint_percentile_threshold=95, 
                embed_model=Settings.embed_model    
            )

        if splitting == "recursive":
            node_parser = LangchainNodeParser(RecursiveCharacterTextSplitter())

        return node_parser

    def _create_extractors(self, extractors: List[str]) -> List:
        """
        Create meta data extractors.
        """
        metadata_extractors = []
        
        if 'summary' in extractors:
                metadata_extractors.append(
                    SummaryExtractor(
                        summaries=["self"],
                        llm=self._llm
                    )
                )
        if 'title' in extractors:
            metadata_extractors.append(
                TitleExtractor(
                    nodes=5,
                    llm=self._llm
                )
            )
        if 'keyword' in extractors:
            metadata_extractors.append(
                KeywordExtractor(
                    keywords=10,
                    llm=self._llm
                )
            )

        return extractors

    def docs_from_urls(self, urls: List[str]) -> List[Document]:
        """
        Get documents from urls.
        """
        documents = SimpleWebPageReader(html_to_text=True).load_data(urls)
        return documents

    def docs_from_dir(self, dir: str) -> List[Document]:
        """
        Get documents from dir.
        """
        documents = SimpleDirectoryReader(input_dir=dir, recursive=True).load_data()
        return documents
    
    def docs_from_github(self, project: str) -> List[Document]:
        """
        Get docs from GitHub repository.
        """
        response = requests.get(f"https://api.github.com/search/repositories?q={project}")
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
                repo=project,
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

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=8,
    )
    def docs_from_web(self, dependency: Dependency, num_websites: int) -> List[Document]:
        """
        Get documents from web using scraping.
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

    def get_vector_store(self, index_name: str):
        return WeaviateVectorStore(
            weaviate_client=self._weaviate_client, 
            index_name=index_name
        )
    
    def index(
        self, 
        index_name: str, 
        documents: List[Document],
        splitting: str,
        extractors: List[str] = [],
        delete_index: bool = True
    ):
        """
        Index data.
        """
        # create and clear vector store
        vector_store = WeaviateVectorStore(
            weaviate_client=self._weaviate_client, 
            index_name=index_name
        )
        if delete_index:
            vector_store.delete_index()

        # build list of transformations
        transformations = [
            self._create_node_parser(splitting=splitting), 
            self._embed_model,
        ]
        transformations.extend(self._create_extractors(extractors=extractors))
        
        # create ingestion pipeline
        pipeline = IngestionPipeline(
            transformations = transformations,
            vector_store = vector_store
        )

        # run ingestion pipeline
        pipeline.run(
            documents=documents,
            show_progress=True
        )



    
      
