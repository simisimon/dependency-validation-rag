from dataclasses import dataclass, field
from llama_index.core.schema import NodeWithScore
from llama_index.core.utils import truncate_text
from typing import Optional, List, Dict
import json
import logging

@dataclass
class Response:
    input: str
    input_complete: str
    response: str
    response_dict: Dict = field(default_factory=dict)
    source_nodes: List[NodeWithScore] = field(default_factory=list)


    def __post_init__(self):
        try:
            self.response_dict = json.loads(self.response)
        except json.JSONDecodeError:
            logging.info("Response cannot be converted into a dict.")

    def __str__(self) -> str:
        """Convert to string representation."""
        return self.response or "None"

    def get_formatted_sources(self, length: int = 100) -> str:
        """Get formatted sources text."""
        texts = []
        for source_node in self.source_nodes:
            fmt_text_chunk = truncate_text(source_node.node.get_content(), length)
            doc_id = source_node.node.node_id or "None"
            source_text = f"> Source (Doc id: {doc_id}): {fmt_text_chunk}"
            texts.append(source_text)
        return "\n\n".join(texts)
    
    def to_dict(self):
        """Convert response object in a dictionary."""
        return {
            "input": self.input,
            "response": self.response,
            "context": [source_node.node.get_content() for source_node in self.source_nodes]
        }
    

@dataclass
class MultipleResponses:
    input: str
    input_complete: str
    responses: List[str]
    response_dict: Dict = field(default_factory=dict)
    source_nodes: List[NodeWithScore] = field(default_factory=list)


    def __str__(self) -> str:
        """Convert to string representation."""
        return json.dumps(self.responses) or "None"

    def get_formatted_sources(self, length: int = 100) -> str:
        """Get formatted sources text."""
        texts = []
        for source_node in self.source_nodes:
            fmt_text_chunk = truncate_text(source_node.node.get_content(), length)
            doc_id = source_node.node.node_id or "None"
            source_text = f"> Source (Doc id: {doc_id}): {fmt_text_chunk}"
            texts.append(source_text)
        return "\n\n".join(texts)
    
    def to_dict(self):
        """Convert response object in a dictionary."""
        return {
            "input": self.input,
            "response": self.responses,
            "context": [source_node.node.get_content() for source_node in self.source_nodes]
        }


@dataclass
class Dependency:
    project: Optional[str] = None
    dependency_type: Optional[str] = None
    dependency_category: Optional[str] = None
    dependency_level: Optional[str] = None 
    option_name: Optional[str] = None
    option_file: Optional[str] = None 
    option_value: Optional[str] = None
    option_type: Optional[str] = None
    option_technology: Optional[str] = None
    dependent_option_name: Optional[str] = None
    dependent_option_value: Optional[str] = None
    dependent_option_type: Optional[str] = None
    dependent_option_file: Optional[str] = None 
    dependent_option_technology: Optional[str] = None

