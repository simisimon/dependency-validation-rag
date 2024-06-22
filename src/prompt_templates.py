from llama_index.core import PromptTemplate


SCRAPING_PROMPT = PromptTemplate(
    "Relationship between '{technologyA}' '{nameA}' and '{technologyB}' '{nameB}'"
)


QUERY_PROMPT = PromptTemplate(
    "{system_str}\n"
    "---------------------\n"
    "Information about both configuration options, including their descriptions and prior usages are stated below:\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information, perform the following task: {task_str}\n\n{format_str}\n\n"
    "Answer:\n {{ “plan”:"
)


SYSTEM_PROMPT = PromptTemplate(
    "You are a full-stack expert in validating intra-technology \
    and cross-technology configuration dependencies.\n" 
    "You will be presented with configuration options found in the software project '{project}'.\n" 
    "Your task is to determine whether the given configuration options \
    actually depend on each other based on value-equality.\n\n"
    "A value-equality dependency is present if two configuration options \
    must have identical values in order to function correctly."
    "Inconsistencies in these configuration values can lead to configuration errors."
    "Importantly, configuration options may have equal values by accident, \
    meaning that there is no actual dependency, but it just happens that they have equal values."
)


TASK_PROMPT = PromptTemplate(
    "Carefully evaluate whether configuration option {nameA} of type {typeA} with value {valueA} in {fileA} of technology {technologyA} \
    depends on configuration option {nameB} of type {typeB} with value {valueB} in {fileB} of technology {technologyB} or vice versa." 
)


FORMAT_STR = """Respond in a JSON format as shown below:
{{
  “plan”: string, // Write down a step-by-step plan on how to solve the task given the information above.
  “rationale”: string, // Provide a concise explanation of whether and why the configuration options depend on each other due to value-equality.
  “uncertainty”: integer, // Rate your certainty of this dependency on a scale from 0 (completely uncertain) to 10 (absolutely certain), given the context, plan, and rationale.
  “isDependency”: boolean // True if a dependency exists, or False otherwise.
}}"""
