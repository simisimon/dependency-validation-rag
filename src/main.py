from dotenv import load_dotenv
from cval import CVal
from data import Dependency


def main():

    dep = Dependency(
        project="piggymetrics",
        dependency_category="value-equality",
        option_name="EXPOSE",
        option_value="8080",
        option_type="PORT",
        option_file="Dockerfile",
        option_technology="Docker",
        dependent_option_name="server.port",
        dependent_option_value="8080",
        dependent_option_file="application.yml",
        dependent_option_type="PORT",
        dependent_option_technology="Spring-Boot"
    )

    config_file = "../config.toml"
    env_file = "../.env"

    cval = CVal.init(
        config_file=config_file,
        env_file=env_file
    )


if __name__ == "__main__":
    main()