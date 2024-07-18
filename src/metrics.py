import sys
sys.path.append('./')

from openai import OpenAI
from eval_models import OpenAIEvalModel, LlamaEvalModel
from dotenv import load_dotenv
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from prompt_settings import PrompSettingsFactory
import pandas as pd
import json
import os
import argparse




def get_model(model_name: str):
    if model_name == "openai":
        model = OpenAI(
            base_url=os.getenv(key="BASE_URL"), 
            api_key=os.getenv(key="PROXY_SERVER_API_KEY")
        )
        return OpenAIEvalModel(model=model)
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_file", type=str, default="../.env")
    parser.add_argument("--data_file", type=str, default="../data/evaluation/config2/all_dependencies_all_gpt-4o-2024-05-13.json")
    parser.add_argument("--model_name", type=str, default="openai", choices=["openai", "llama3"])
    parser.add_argument("--tool_name", type=str, default="cfgnet")

    return parser.parse_args()


def compute_context_relevancy(args):
    print("Compute context relevancy.")

    context_relevancy_metric = ContextualRelevancyMetric(
        threshold=0.7,
        model=get_model(model_name=args.model_name),
        include_reason=True
    )  

    prompt_settings = PrompSettingsFactory().get_prompt_settings(tool_name=args.tool_name)

    with open(args.data_file, "r", encoding="utf-8") as src:
        data = json.load(src)


    for entry in data[:1]:  
        #if "context_relevance" in entry:
        #    continue

        #context = [x["content"] for x in entry["context"]]
        #print(context)
        #print(len(context))
        #print(f"{entry['task_str']}\n\n{prompt_settings.get_format_prompt()}",)

        test_case = LLMTestCase(
            input=f"{entry['task_str']}\n\n{prompt_settings.get_format_prompt()}",
            actual_output=entry["response"],
            retrieval_context=[entry["context_str"]]
        )

        context_relevancy_metric.measure(test_case)
        print("Context Relevancy Score: ", context_relevancy_metric.score)
        print("Context Relevancy Reason: ", context_relevancy_metric.reason)

        #entry["context_relevance"] = context_relevancy_metric.score

    #with open(args.data_file, "w", encoding="utf-8") as dest:
    #    json.dump(data, dest, indent=2)


def compute_answer_relevancy(args):
    print("Compute answer relevancy.")

    answer_relevancy_metric = AnswerRelevancyMetric(
        threshold=0.7,
        model=get_model(model_name=args.model_name),
        include_reason=True
    )

    prompt_settings = PrompSettingsFactory().get_prompt_settings(tool_name=args.tool_name)

    with open(args.data_file, "r", encoding="utf-8") as src:
        data = json.load(src)


    for entry in data[:1]:  
        #if "context_relevance" in entry:
        #    continue

        #context = [x["content"] for x in entry["context"]]
        #print(context)
        #print(len(context))
        #print(f"{entry['task_str']}\n\n{prompt_settings.get_format_prompt()}",)

        test_case = LLMTestCase(
            input=f"{entry['task_str']}\n\n{prompt_settings.get_format_prompt()}",
            actual_output=entry["response"],
            retrieval_context=[entry["context_str"]]
        )

        answer_relevancy_metric.measure(test_case)
        print("Answer Relevancy: ", answer_relevancy_metric.score)
        print("Answer Relevancy: ", answer_relevancy_metric.reason)

        #entry["context_relevance"] = context_relevancy_metric.score

    #with open(args.data_file, "w", encoding="utf-8") as dest:
    #    json.dump(data, dest, indent=2)


def compute_faithfulness(args):
    print("Compute faithfulness.")

    faithfulness_metric = FaithfulnessMetric(
        threshold=0.7,
        model=get_model(model_name=args.model_name),
        include_reason=True
    )

    prompt_settings = PrompSettingsFactory().get_prompt_settings(tool_name=args.tool_name)

    with open(args.data_file, "r", encoding="utf-8") as src:
        data = json.load(src)


    for entry in data[:1]:  
        #if "context_relevance" in entry:
        #    continue

        #context = [x["content"] for x in entry["context"]]
        #print(context)
        #print(len(context))
        #print(f"{entry['task_str']}\n\n{prompt_settings.get_format_prompt()}",)

        test_case = LLMTestCase(
            input=f"{entry['task_str']}\n\n{prompt_settings.get_format_prompt()}",
            actual_output=entry["response"],
            retrieval_context=[entry["context_str"]]
        )

        faithfulness_metric.measure(test_case)
        print("Faithfulness Score: ", faithfulness_metric.score)
        print("Faithfulness Reason: ", faithfulness_metric.reason)

        #entry["context_relevance"] = context_relevancy_metric.score

    #with open(args.data_file, "w", encoding="utf-8") as dest:
    #    json.dump(data, dest, indent=2)


def main():
    args = get_args()

    load_dotenv(dotenv_path=args.env_file)  

    #compute_context_relevancy(args=args)
    #compute_answer_relevancy(args=args)
    compute_faithfulness(args=args)

if __name__ == "__main__":
    main()