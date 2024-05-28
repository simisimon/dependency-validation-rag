from generator import GeneratorFactory
from dotenv import dotenv_values
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable_rag", type=bool, default=False)
    parser.add_argument("--context_type", type=str, choices=["stack_overflow", "documentation", "blog_posts"])
    parser.add_argument("--model_name", type=str, default="llama3:70b", choices=["gpt-4-0125-preview", "gpt-3.5-turbo-0125", "llama3:8b", "gpt-4o-2024-05-13"])
    parser.add_argument("--temperature", type=float, default=0.0, choices=[0.0, 0.2, 0.7, 1.0])
   
    return parser.parse_args()


def main():
    args = get_args()

    config = dotenv_values("../.env")

    print(config["OPENAI_KEY"])


    # TODO: get dependencies from software project
    # TODO: get context information
    # TODO: create prompt
    # TODO: utilize LLMS
    # TODO: store results


if __name__ == "__main__":
    main()