from dataclasses import dataclass, field
from llama_index.core.schema import NodeWithScore
from llama_index.core.utils import truncate_text
from typing import Optional, List

@dataclass
class Response:
    input: str
    response: str
    source_nodes: List[NodeWithScore] = field(default_factory=list)
    expected_output: str = None
    context: List[str] = None

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

