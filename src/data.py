from dataclasses import dataclass
from typing import Optional


@dataclass
class CvalConfig:
    enable_rag: Optional[bool] = None
    env_file_path: Optional[str] = None
    index_name: Optional[str] = None
    top_k: Optional[int] = None
    retrieval_type: Optional[str] = None
    model_name: Optional[str] = None
    temperature: Optional[float] = None
    num_websites: Optional[int] = None
    dimension: Optional[int] = None
    metrics: Optional[str] = None


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

