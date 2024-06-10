from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, List
from pydantic import BaseModel


class ValidationResponse(BaseModel):
    """Data model for validation response."""
    explanation: str
    uncertainty: int
    isDependency: bool


class DependencyLevel(Enum):
    CONFIG_FILE_LEVEL = auto()
    CODE_LEVEL = auto()


class DependencyType(Enum):
    INTRA = auto()
    CROSS = auto()


class DependencyCategory(Enum):
    VALUE = auto()
    CONTROL = auto()
    BEHAVIORAL = auto()
    OVERWRITE = auto()


@dataclass
class Dependency:
    # Required args
    project: str
    dependency_type: str
    dependency_category: str
    dependency_level: str 
    option_name: str
    option_file: str 
    option_technology: str
    dependent_option_name: str
    dependent_option_file: str 
    dependent_option_technology: str

    # Optional args
    option_value: Optional[str] = None
    option_type: Optional[str] = None
    dependent_option_value: Optional[str] = None
    dependent_option_type: Optional[str] = None
    method_name: Optional[str] = None
    class_name: Optional[str] = None
    method_body: Optional[str] = None


    def __str__(self) -> str:
        if self.dependency_level == DependencyLevel.CONFIG_FILE_LEVEL.name:
            dependency_str = (f"There is a potential {self.dependency_type.lower()}-technology dependency between the following configuration options found in {self.project}:\n"
            f"- Option A: The configuration option named {self.option_name} with the value {self.option_value} of type {self.option_type} located in file {self.option_file} from the technology {self.option_technology}.\n"
            f"- Option B: The configuration option named {self.dependent_option_name} with the value {self.dependent_option_value} of type {self.dependent_option_type} located in file {self.dependent_option_file} from the technology {self.option_technology}.\n"
            f"Both configuration options may depend on each other due to a {self.dependency_category} dependency.")
        elif self.dependency_level == DependencyLevel.CODE_LEVEL.name:
            dependency_str = (f"There is a potential {self.dependency_type.lower()}-technology dependency between the following configuration options found in {self.project}:\n"
            f"- Option A: The configuration option named {self.option_name} located in the method {self.method_name} of the class {self.class_name} in file {self.option_file}.\n"
            f"- Option B: The configuration option named {self.dependent_option_name} located in the method {self.method_name} of the class {self.class_name} in file {self.option_file}..\n"
            f"Both configuration options may depend on each other due to a {self.dependency_category} dependency that manifests in the following code snippet:\n."
            f"{self.method_body}")
        else:
            dependency_str = (f"There is a potential {self.dependency_type.lower()}-technology dependency between the following configuration options found in {self.project}:\n"
            f"- Option A: The configuration option named {self.option_name} located in file {self.option_file} from the technology {self.option_technology}.\n"
            f"- Option B: The configuration option named {self.dependent_option_name} located in file {self.dependent_option_file} from the technology {self.dependent_option_technology}.\n"
            f"Both configuration options may depend on each other due to a {self.dependency_category} dependency.")

        return dependency_str