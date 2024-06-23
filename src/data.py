from dataclasses import dataclass
from typing import Optional


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

