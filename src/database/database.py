from pinecone import Pinecone, ServerlessSpec
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from dotenv import load_dotenv
from typing import List, Dict
import os

class VectorDatabase:
    def __init__(self) -> None:
        self.instance = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    def _set_embed_model(self, embed_model: str):

        if embed_model == "openai":
            Settings.embed_model = OpenAIEmbedding(api_key=os.getenv("OPENAI_KEY"))
        if embed_model == "huggingface":
            raise NotImplementedError()


    def get_index(self, args: Dict):
        if args.index_name not in self.instance.list_indexes().names():
            raise Exception(f"Index {args.index_name} does not exist.")
        
        index = self.instance.Index(args.index_name)

        return index

    def create_index(self, args: Dict) -> None:
        """Create a new index."""
        if args.index_name not in self.instance.list_indexes().names():
            print(f"Create index for {args.index_name}")
            self.instance.create_index(
                name=args.index_name,
                dimension=args.dimension,
                metric=args.metric,
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        
    
    def add_documents(self, args: Dict) -> None:
        pass


    def get_retrieval_engine(self, args: Dict):
        if args.index_name not in self.instance.list_indexes().names():
            raise Exception(f"Index {args.index_name} does not exist.")
        
        # set query engine settings
        self._set_embed_model(embed_model=args.embed_model)

        index = self.instance.Index(args.index_name)
        vector_store = PineconeVectorStore(pinecone_index=index)
        vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

        return VectorIndexRetriever(index=vector_index, similarity_top_k=5)



    
      
