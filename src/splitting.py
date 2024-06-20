from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter, SemanticSplitterNodeParser, LangchainNodeParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core import Settings


class DataSplittingFactory:
    @staticmethod
    def get_splitting(splitting: str):
        """
        Get splitting method.
        """
        splitter = None

        if splitting == "token":
            splitter = TokenTextSplitter(
                chunk_size=512,
                chunk_overlap=50,
                separator=" ",
            )

        if splitting == "sentence":
            splitter = SentenceSplitter(
                chunk_size=512, 
                chunk_overlap=50
            )

        if splitting == "semantic":
            splitter = SemanticSplitterNodeParser(
                buffer_size=1, 
                breakpoint_percentile_threshold=95, 
                embed_model=Settings.embed_model    
            )

        if splitting == "recursive":
            splitter = LangchainNodeParser(RecursiveCharacterTextSplitter())

        return splitter
