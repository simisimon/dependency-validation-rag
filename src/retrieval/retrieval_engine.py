from pinecone import Pinecone, ServerlessSpec
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from dotenv import load_dotenv
from typing import List, Dict
import os
import re


class RetrievalEngine:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.instance = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.set_settings(
            embed_model=self.config["embed_model"]
        )

    def set_settings(self, embed_model: str):
        if embed_model == "openai":
            Settings.embed_model = OpenAIEmbedding(api_key=os.getenv("OPENAI_KEY"))
        if embed_model == "huggingface":
            raise NotImplementedError()

    def get_index(self, index_name: str):
        """
        Get Index.
        """
        if index_name not in self.instance.list_indexes().names():
            raise Exception(f"Index {index_name} does not exist.")
        
        index = self.instance.Index(index_name)

        return index
    
    def get_vector_store(self, index):
        """
        Get VectorStore.
        """
        vector_store = PineconeVectorStore(pinecone_index=index)
        return vector_store

    
    def get_vector_index(self, vector_store):
        """
        Get VectorStoreIndex.
        """
        raise NotImplementedError()



    def add_documents(self, index_name: str, document_dir: str) -> None:
        """
        Add documents to index. If index does not exist, create a new index.
        
        :param index_name: Name of index.
        :param documents: Path to documents to be indexed.    
        """
        
        if index_name not in self.instance.list_indexes().names():
            print(f"Create index for {index_name}")
            self.instance.create_index(
                name=index_name,
                dimension=self.config["dimension"],
                metric=self.config["metric"],
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

        index = self.instance.Index(index_name)
        vector_store = PineconeVectorStore(pinecone_index=index)

        cleaned_documents = []

        documents = SimpleDirectoryReader(dir).load_data()

        for document in documents: 
            cleaned_text = self._clean_up_text(document.text)
            document.text = cleaned_text
            cleaned_documents.append(document)

        pipeline = IngestionPipeline(
            transformations=[
                SemanticSplitterNodeParser(
                    buffer_size=1,
                    breakpoint_percentile_threshold=95, 
                    embed_model=Settings.embed_model,
                ),
                Settings.embed_model,
            ],
            vector_store=vector_store
        )

        pipeline.run(documents=cleaned_documents)


    
    def _clean_up_text(content: str) -> str:
        """
        Remove unwanted characters and patterns in text input.

        :param content: Text input.
        
        :return: Cleaned version of original text input.
        """

        # Fix hyphenated words broken by newline
        content = re.sub(r'(\w+)-\n(\w+)', r'\1\2', content)

        # Remove specific unwanted patterns and characters
        unwanted_patterns = [
            "\\n", "  —", "——————————", "—————————", "—————",
            r'\\u[\dA-Fa-f]{4}', r'\uf075', r'\uf0b7'
        ]
        for pattern in unwanted_patterns:
            content = re.sub(pattern, "", content)

        # Fix improperly spaced hyphenated words and normalize whitespace
        content = re.sub(r'(\w)\s*-\s*(\w)', r'\1-\2', content)
        content = re.sub(r'\s+', ' ', content)

        return content
    


    
      
