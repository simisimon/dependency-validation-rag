from fake_useragent import UserAgent
from bs4 import BeautifulSoup
from typing import List
from rich.logging import RichHandler
from xhtml2pdf import pisa
from io import BytesIO
import backoff
import requests
import logging
import re
import os


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()],
)


HTML_CONTENT = """
<!DOCTYPE html>
<html>
<body>
    {}
</body>
</html>
"""


class ScrapingException(Exception):
    def __init__(self):            
        super().__init__()


class Scraper:
    def __init__(self) -> None:
        self.base_url = "https://www.bing.com/search?q="

    @backoff.on_exception(
        backoff.expo,
        ScrapingException,
        max_tries=8,
    )
    def scrape(self, output_dir: str, query: str, num_documents: int):
        
        url = self.base_url + query
        urls = self.get_urls(url=url)

        if not urls or len(urls) < num_documents: 
            raise ScrapingException()
        
        for url in urls[:num_documents]:
            logging.info(f"Scrape url: {url}")
            doc_name = url.split("/")[-1]

            if not doc_name:
                doc_name = "-".join(url.split("/")[1:])

            output_file = f"{output_dir}/{doc_name}.pdf"

            if not os.path.exists(output_file):
                logging.info("File already exists")
                
            response = requests.get(url, headers={'User-Agent':  UserAgent().chrome})
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            for tag in soup(["script", "style", "img"]):  
                tag.decompose() 

            pdf_output = BytesIO()
            pisa.CreatePDF(HTML_CONTENT.format(soup), dest=pdf_output, encoding='utf-8')

            with open(output_file, "wb") as pdf_file:
                pdf_file.write(pdf_output.getvalue())
    
    def get_urls(self, url: str) -> List[str]:
        urls = []

        response = requests.get(url, headers={'User-Agent':  UserAgent().chrome})
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        search_urls = soup.find_all("h2")
        search_urls = [result for result in search_urls if "https://" in str(result)]

        for elem in search_urls:
            match = re.findall(r'href="([^"]*)"', str(elem))
            link = match[0]
            urls.append(link)
        
        return urls