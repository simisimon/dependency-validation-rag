from prompt_settings import PrompSettingsFactory
from util import load_config
from dotenv import load_dotenv
from generator import GeneratorFactory
from typing import List, Dict
from tqdm import tqdm
from generator import GeneratorEngine
import mlflow
import json
import argparse
import json
import backoff


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="../generation_config.toml")
    parser.add_argument("--env_file", type=str, default="../.env")   
    
    return parser.parse_args()


@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def generate(generator: GeneratorEngine, messages: List) -> str:
    response = generator.generate(messages=messages)

    if not response:
        raise Exception("Response is empty.")
    
    try:
        response_dict = json.loads(response)
        if "isDependency" not in response_dict:
            raise Exception("KeyError: isDependency")
    except json.JSONDecodeError:
        raise Exception("Response format not serializable.")

    return response


def run_generation(config: Dict) -> None:
    
    print("Data file: ", config["data_file"])
    print("With RAG?: ", config["with_rag"])

    with open(config["data_file"], "r", encoding="utf-8") as src:
        data = json.load(src)

    prompt_settings = PrompSettingsFactory.get_prompt_settings(tool_name=config["tool_name"])

    results = []
    batch_size = 100
    counter = 0

    generator = GeneratorFactory().get_generator(
        model_name=config["model_name"], 
        temperature=config["temperature"]
    )

    for entry in tqdm(data, total=len(data), desc="Processing entries"):

        if config["with_rag"]:
            query_str = prompt_settings.query_prompt.format(
                context_str=entry["context_str"], 
                task_str=entry["task_str"],
                format_str=prompt_settings.get_format_prompt()
            )
        else:
            query_str =f"{entry['task_str']}\n\n{prompt_settings.get_format_prompt()}"

        messages = [
            {
                "role": "system", 
                "content": entry["system_str"]
            },
            {
                "role": "user",
                "content": query_str
            }
        ]

        try:
            response = generate(
                generator=generator,
                messages=messages
            )
        except Exception:
            response = "None"

        entry["response"] = response
        counter += 1
        results.append(entry)

        if counter % batch_size == 0:
            if config["with_rag"]:
                output_file = f"{config['output_dir']}/all_dependencies_{config['index_name']}_{config['model_name']}_{counter}.json"
            else:
                output_file = f"{config['output_dir']}/all_dependencies_without_{config['model_name']}_{counter}.json"
            with open(output_file, "a", encoding="utf-8") as dest:
                json.dump(results, dest, indent=2)
            mlflow.log_artifact(local_path=output_file) 


def main():
    args = get_args()

    # load env variable
    load_dotenv(dotenv_path=args.env_file)

    # load config
    config = load_config(config_file=args.config_file)

    mlflow.set_experiment(experiment_name="generation")
    
    with mlflow.start_run(run_name=f"generation_{config['index_name']}"): 

        mlflow.log_params(config)
        mlflow.log_artifact(local_path=config["data_file"])
        mlflow.log_artifact(local_path=args.env_file)

        run_generation(config=config)


if __name__ == "__main__":
    main()









