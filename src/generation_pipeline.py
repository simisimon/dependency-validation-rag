from prompt_settings import PrompSettingsFactory, AdvancedCfgNetPromptSettings
from util import load_config, load_shots, get_most_similar_shots, get_projet_description, transform
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
import pandas as pd


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
        response_dict = json.loads(response, strict=False)
        if "isDependency" not in response_dict:
            raise Exception("KeyError: isDependency")
    except json.JSONDecodeError:
        raise Exception("Response format not serializable.")

    return response


def run_generation(config: Dict) -> None:
    
    print(f"Run generation for {config['data_file']} with RAG {config['with_rag']}")

    with open(config["data_file"], "r", encoding="utf-8") as src:
        data = json.load(src)

    prompt_settings = PrompSettingsFactory.get_prompt_settings(tool_name="cfgnet")

    results = []

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
            {"role": "system", "content": entry["system_str"]},
            {"role": "user", "content": query_str}
        ]

        try:
            response = generate(
                generator=generator,
                messages=messages
            )
        except Exception:
            response = "None"

        entry["response"] = response
        results.append(entry)

    output_file = config["output_file"]
    with open(output_file, "w", encoding="utf-8") as dest:
        json.dump(results, dest, indent=2)


def run_advanced_generation(config: Dict) -> None:
    
    print(f"Run generation with refinements for {config['data_file']} with RAG {config['with_rag']}")

    with open(config["data_file"], "r", encoding="utf-8") as src:
        data = json.load(src)

    prompt_settings = AdvancedCfgNetPromptSettings

    results = []
    shots = load_shots()

    generator = GeneratorFactory().get_generator(
        model_name=config["model_name"], 
        temperature=config["temperature"]
    )

    for entry in tqdm(data, total=len(data), desc="Processing entries"):

        dependency = transform(entry["dependency"])

        project_str = get_projet_description(project_name=dependency.project)
        context_str = entry["context_str"]
        task_str = prompt_settings.get_task_str(dependency=dependency)
        shots_str = "\n\n".join([shot for shot in get_most_similar_shots(shots, dependency)])
        format_str = prompt_settings.get_format_prompt()

        system_prompt = prompt_settings.get_system_str(
            dependency=dependency,
            project_str=project_str
        )

        user_prompt = prompt_settings.advanced_query_prompt.format(
                context_str=context_str, 
                shot_str=shots_str,
                task_str=task_str,
                format_str=format_str
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = generate(
                generator=generator,
                messages=messages
            )
        except Exception:
            response = "None"

        entry["response"] = response
        
        results.append(entry)

        
    output_file = config["output_file"]
    with open(output_file, "a", encoding="utf-8") as dest:
        json.dump(results, dest, indent=2)


def main():
    args = get_args()

    # load env variable
    load_dotenv(dotenv_path=args.env_file)

    # load config
    config = load_config(config_file=args.config_file)

    mlflow.set_experiment(experiment_name="generation")
    
    with mlflow.start_run(run_name=f"generation_{config['model_name']}"): 

        mlflow.log_params(config)
        mlflow.log_artifact(local_path=config["data_file"])
        mlflow.log_artifact(local_path=args.env_file)

        if not config["with_refinements"]:
            run_generation(config=config)
        else:
            run_advanced_generation(config=config)


if __name__ == "__main__":
    main()









