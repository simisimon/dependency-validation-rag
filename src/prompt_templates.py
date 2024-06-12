from llama_index.core import PromptTemplate


SCRAPING_PROMPT = PromptTemplate(
    "Relationship between '{technologyA}' '{nameA}' and '{technologyB}' '{nameB}'"
)


QUERY_PROMPT = PromptTemplate(
    "{system_str}\n"
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information, perform the following task: {task_str}\n\n{format_str}"
)


SYSTEM_PROMPT = PromptTemplate(
    "You are an expert in validating intra-technology and cross-technology configuration dependencies.\n" 
    "You will be presented with configuration options found in the software project '{project}'.\n" 
    "Your task is to determine whether the given configuration options actually depend on each other based on value-equality.\n\n"
    "{definition_str}"
)


VALUE_EQUALITY_DEFINITION_STR = """A value equality dependency specifies that certain configuration options must have identical values in order to function correctly.
Inconsistencies in these configuration values can lead to dependency conflicts of varying severity.
However, configuration dependencies based on value equality carry the risk of being false positives, as configuration options whose values are equal do not necessarily have to depend on each other."""


TASK_PROMPT = PromptTemplate(
    "Carefully verify whether configuration option {nameA} of type {typeA} with value {valueA} in {fileA} of technology {technologyA} depends on configuration option {nameB} of type {typeB} with value {valueB} in {fileB} of technology {technologyB} or vice versa." 
)


FORMAT_STR = """Respond in a JSON format as shown below:
{{
  "rationale": string, // Provide a concise explanation of whether and why the configuration options depend on each other due to value-equality.
  "uncertainty": integer, // Rate your certainty of this dependency on a scale from 0 (completely uncertain) to 10 (absolutely certain).
  "isDependency": boolean // True if a dependency exists, or False otherwise.
}}"""


QUERY_GEN_PROMPT = PromptTemplate(
    "You are a helpful assistant that generates multiple search queries based on a single input query.\n"
    "Generate {num_queries} search queries, one on each line, related to the following input query: {query}"
)