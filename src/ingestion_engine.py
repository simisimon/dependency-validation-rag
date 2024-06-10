from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.core import Settings
from llama_index.core.ingestion import IngestionPipeline
from llama_index.readers.web import SimpleWebPageReader
from typing import List, Optional
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
import re
import backoff
import requests


class ScrapingException(Exception):
    def __init__(self):            
        super().__init__()


class DataIngestionEngine:
    def __init__(self) -> None:
        self.base_url = "https://www.bing.com/search?q="
    
    @backoff.on_exception(
        backoff.expo,
        ScrapingException,
        max_tries=8,
    )
    def scrape(
        self, 
        query: str,
        website: Optional[str],
        num_websites: int = 5) -> List:
        """
        Scrape websites and return list of corresponding documents.
        """
        if not website:
            url = self.base_url + query
        else:
            url = self.base_url + "site:" + website + " " + query

        response = requests.get(
            url, 
            headers={'User-Agent':  UserAgent().chrome}
        )
        response.raise_for_status()
        
        search_urls = []

        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all("h2")

        if not website:
            links = [x for x in links if "https://" in str(x)]
        else:
            links = [x for x in links if "https://" in str(x) and website in str(x)]

        for link in links:
            match = re.findall(r'href="([^"]*)"', str(link))
            search_urls.append(match[0])

        if not search_urls or len(search_urls) < num_websites:
            raise ScrapingException() 
        
        print("Links to scrape: ", search_urls)

        documents = SimpleWebPageReader(html_to_text=True).load_data(search_urls)

        return documents
    

    def index_documents(
            self,
            vector_store,
            documents: List,
        ) -> None:
        """
        Add documents to a vector store
        """
        transformations = [
            SentenceSplitter(chunk_size=1024, chunk_overlap=20),
            Settings.embed_model,
        ]

        pipeline = IngestionPipeline(
            transformations=transformations,
            vector_store=vector_store
        )

        print(f"Start indexing.")
        pipeline.run(documents=documents)
        print(f"Indexing done.")


    
      
