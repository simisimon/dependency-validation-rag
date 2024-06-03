from dataclasses import dataclass
from typing import List
from cfgnet.network.network import Network, NetworkConfiguration
import argparse


@dataclass
class Dependency:
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


def transform(links: set, num_dependencies: int) -> List:
    pass


def get_args():
    parser = argparse.ArgumentParser()
    
    # CfgNet config
    parser.add_argument("--repo_path", type=str)
    parser.add_argument("--num_dependencies", type=int, default=10)

    return parser.parse_args()

def main():
    args = get_args()

    cfg = NetworkConfiguration(
        project_root_abs=args.repo_pat
        enable_internal_links=True,
        enable_static_blacklist=True
    )

    network = Network().init_network(cfg=cfg)
    links = network.links
    dependencies = transform(links=links, num_dependencies=args.num_dependencies)

    


if __name__ == "__main__":
    main()