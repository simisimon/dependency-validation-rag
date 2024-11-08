from llama_index.core import Settings, SimpleDirectoryReader, Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.readers.web import SimpleWebPageReader
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.extractors import SummaryExtractor, TitleExtractor, KeywordExtractor
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter, SemanticSplitterNodeParser, LangchainNodeParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from typing import List
from rich.logging import RichHandler
from duckduckgo_search import DDGS
import traceback
import backoff
import requests
import logging
import os
import nest_asyncio


nest_asyncio.apply()


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()],
)


class IngestionEngine:
    def __init__(
        self, 
        pinecone_client: Pinecone, 
        dimension: int, 
        splitting: str, 
        extractors: List[str]
    ) -> None:
        logging.info(f"Ingestion engine initialized.")
        self._pinecone_client = pinecone_client
        self.dimension = dimension
        self.splitting = splitting
        self.extractors = extractors


    def _get_vector_store(self, index_name: str) -> PineconeVectorStore:
        """
        Get Pinecone vector store. If index does not exist, create index.
        """
        if index_name not in self._pinecone_client.list_indexes().names():
            logging.info(f"Create Index {index_name}.")
            self._pinecone_client.create_index(
                name=index_name,
                dimension=self.dimension,
                metric="dotproduct",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

        index = self._pinecone_client.Index(index_name)
        vector_store = PineconeVectorStore(
            pinecone_index=index,
            add_sparse_vector=True
        )
        return vector_store


    def _create_node_parser(self):
        node_parser = None

        if self.splitting == "token":
            logging.info("Set token splitting.")
            node_parser = TokenTextSplitter(
                chunk_size=512,
                chunk_overlap=50,
                separator=" ",
            )

        if self.splitting == "sentence":
            logging.info("Set sentence splitting.")
            node_parser = SentenceSplitter(
                chunk_size=512, 
                chunk_overlap=50
            )

        if self.splitting == "semantic":
            logging.info("Set semantic splitting.")
            node_parser = SemanticSplitterNodeParser(
                buffer_size=5, 
                breakpoint_percentile_threshold=80, 
                embed_model=Settings.embed_model    
            )

        if self.splitting == "recursive":
            node_parser = LangchainNodeParser(RecursiveCharacterTextSplitter())

        return node_parser
    

    def _create_extractors(self) -> List:
        """
        Create meta data extractors.
        """
        metadata_extractors = []
            
        if "summary" in self.extractors:
                metadata_extractors.append(
                    SummaryExtractor(
                        summaries=["self"],
                        llm=Settings.llm
                    )
                )
        if "title" in self.extractors:
            metadata_extractors.append(
                TitleExtractor(
                    nodes=5,
                    llm=Settings.llm
                )
            )
        if "keyword" in self.extractors:
            metadata_extractors.append(
                KeywordExtractor(
                    keywords=10,
                    llm=Settings.llm
                )
            )

        return metadata_extractors


    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=10,
    )
    def docs_from_web(self, query_str: str, num_websites: int) -> List[Document]:
        """
        Get documents from websites.
        """
        results = DDGS().text(query_str, max_results=num_websites)
        urls = []
        for result in results:
            url = result['href']
            urls.append(url)
 
        documents = SimpleWebPageReader(html_to_text=True).load_data(urls)

        return documents
    

    def docs_from_github(self, project_name: str) -> List[Document]:
        """
        Get documents from GitHub repository.
        """
        logging.info(f"Start scraping the repository of {project_name}.")
        response = requests.get(f"https://api.github.com/search/repositories?q={project_name}")
        response.raise_for_status()

        data = response.json()

        if data['total_count'] > 0:
            owner = data["items"][0]["owner"]["login"]
            branch = data["items"][0]["default_branch"]
            repo_name = data['items'][0]["name"]
        else:
            return []
        
        print(owner)
        print(branch)
                
        try:
            github_client = GithubClient(
                github_token=os.getenv(key="GITHUB_TOKEN"), 
                verbose=True
            )

            documents = GithubRepositoryReader(
                github_client=github_client,
                owner=owner,
                repo=repo_name,
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
                        ".conf",
                        ".md"
                    ],
                    GithubRepositoryReader.FilterType.INCLUDE,
                ),
            ).load_data(branch=branch)    
            return documents
        except Exception:
            logging.info("Error occurred while scraping Github.")
            print(traceback.format_exc)
            return []


    def docs_from_dir(self, data_dir: str) -> List[Document]:
        """
        Get documents from data directory.
        """
        documents = SimpleDirectoryReader(input_dir=data_dir, recursive=True).load_data()
        return documents
    

    def docs_from_urls(self, urls: List[str]) -> List[Document]:
        """
        Get documents from urls.
        """
        documents = SimpleWebPageReader(html_to_text=True).load_data(urls)
        return documents
    

    def index_documents(
            self,
            index_name: str,
            documents: List,
            delete_index: bool,
    ) -> None:
        """
        Add documents to a vector store
        """
        # delete index
        if delete_index:
            if index_name in self._pinecone_client.list_indexes().names():
                self._pinecone_client.delete_index(name=index_name)
                logging.info(f"Delete index {index_name}")

        # create pinecone vector store
        vector_store = self._get_vector_store(index_name=index_name)

        # create node splitting parser
        node_parser = self._create_node_parser()

        # create meta data extractors
        extractors = self._create_extractors()

        # build list of transformations
        transformations = [node_parser, Settings.embed_model]
        transformations.extend(extractors)

        # create ingestion pipeline
        pipeline = IngestionPipeline(
            transformations=transformations,
            vector_store=vector_store
        )

        # run ingestion pipeline
        pipeline.run(
            documents=documents,
            show_progress=True
        )
    


    
      
