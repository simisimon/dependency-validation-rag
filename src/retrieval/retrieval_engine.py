from pinecone import Pinecone, ServerlessSpec
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex, ServiceContext, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from rich.logging import RichHandler
from typing import Tuple
from dotenv import load_dotenv
import os
import re
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()],
)


class RetrievalEngine:
    def __init__(self, embed_model_name: str) -> None:
        self.instance = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.set_settings(embed_model_name=embed_model_name)

    def set_settings(self, embed_model_name: str):
        load_dotenv()
        if embed_model_name == "openai":
            Settings.embed_model = OpenAIEmbedding(api_key=os.getenv("OPENAI_KEY"))
        if embed_model_name == "huggingface":
            raise NotImplementedError()

    def retrieve_context(self, query: str, index_name: str, top_k: int) -> Tuple:
        """
        Retrieve relevant context from index.
        """
        if index_name not in self.instance.list_indexes().names():
            raise Exception(f"Index {index_name} does not exist.")
        
        index = self.instance.Index(index_name)
        vector_store = PineconeVectorStore(pinecone_index=index)

        service_context = ServiceContext.from_defaults(
            llm=None, 
            embed_model=Settings.embed_model
        )

        vector_index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, 
            service_context=service_context
        )

        retriever = VectorIndexRetriever(
            index=vector_index, 
            similarity_top_k=top_k
        )


        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)]
        )

        query_results = query_engine.query(query)

        return query_results


    def add_documents(self, index_name: str, document_dir: str, dimension: int, metric: str) -> None:
        """
        Add documents to index. If index does not exist, create a new index.
        
        :param index_name: Name of index.
        :param documents: Path to documents to be indexed.    
        """
        if index_name not in self.instance.list_indexes().names():
            logging.info(f"Create index: {index_name}.")
            self.instance.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

        index = self.instance.Index(index_name)
        vector_store = PineconeVectorStore(pinecone_index=index)

        cleaned_documents = []

        documents = SimpleDirectoryReader(document_dir).load_data()

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

        logging.info(f"Start indexing.")
        pipeline.run(documents=cleaned_documents)
        logging.info(f"Indexing done.")


    def _clean_up_text(self, content: str) -> str:
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
    


    
      
