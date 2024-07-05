from data import Dependency
from cval import CVal
import mlflow
import pandas as pd
import json
import glob
import os


CONFIG_FILE = "../config.toml"
ENV_FILE = "../.env"
EVAL_DATA_DIR = "../data/evaluation/data"
INDEX_NAME = "without"
EVAL_FILE_PATH = "../data/evaluation/data/apollo_dependencies.csv"

def run_inference(file_path):

    file_name = file_path.split("/")[-1].split(".")[0]
    print("File: ", file_name)

    if os.path.exists(f"../data/evaluation/results/{file_name}_{INDEX_NAME}.json"):
        print(f"{file_name}_{INDEX_NAME}.json already exists. Skip file.")
        return

    with mlflow.start_run(run_name=f"{file_name}_{INDEX_NAME}"): 

        cval = CVal.init(
            config_file=CONFIG_FILE,
            env_file=ENV_FILE
        )

        df = pd.read_csv(file_path)

        outputs = []
        for x in df.to_dict("records"):
            dependency = Dependency(
                project=x["project"],
                option_name=x["option_name"],
                option_value=x["option_value"],
                option_type=x["option_type"].split(".")[-1],
                option_file=x["option_file"],
                option_technology=x["option_technology"],
                dependent_option_name=x["dependent_option_name"],
                dependent_option_value=x["dependent_option_value"],
                dependent_option_type=x["dependent_option_type"].split(".")[-1],
                dependent_option_file=x["dependent_option_file"],
                dependent_option_technology=x["dependent_option_technology"]
            )

            response = cval.query(
                dependency=dependency,
                index_name=INDEX_NAME
            )


            outputs.append(response)

        responses = [response.to_dict() for response in outputs]

        with open(f"../data/evaluation/results/{file_name}_{INDEX_NAME}.json", "w", encoding="utf-8") as dest:
            json.dump(responses, dest, indent=2)

        mlflow.log_artifact(local_path=f"../data/evaluation/results/{file_name}_{INDEX_NAME}.json")

        print("Done with: ", file_path)



def main():
    

    mlflow.set_experiment(experiment_name=f"inference_{INDEX_NAME}")

    #for file_path in glob.glob(EVAL_DATA_DIR + "/**"):
    run_inference(file_path=EVAL_FILE_PATH)


if __name__ == "__main__":
    main()



