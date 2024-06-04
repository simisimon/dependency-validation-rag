from dataclasses import dataclass

@dataclass
class Dependency:
    project: str = None
    link: str = None
    dependency_category: str = None

    option_name: str = None
    option_value: str = None
    option_type: str = None
    option_file: str = None
    option_technology: str = None

    dependent_option_name: str = None
    dependent_option_value: str = None
    dependent_option_type: str = None
    dependent_option_file: str = None
    dependent_option_technology: str = None