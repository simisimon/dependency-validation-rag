from llama_index.core import PromptTemplate


RETRIEVAL_STR = PromptTemplate(
    "Are the following configuration options found in {project} dependent on each other?\n"
    "Option A: The configuration option named '{nameA}' with the value '{valueA}' of type '{typeA}' located in '{fileA}', which belongs to the technology '{technologyA}'.\n"
    "Option B: The configuration option named '{nameB}' with the value '{valueB}' of type '{typeB}' located in '{fileB}', which belongs to the technology '{technologyB}'.\n\n"
)


QUERY_PROMPT = PromptTemplate(
    "{system_str}\n"
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information, perform the following task.\n"
    "---------------------\n"
    "{task_str}\n"
)


SYSTEM_STR = PromptTemplate(
    "You are an expert in validating intra-technology and cross-technology configuration dependencies.\n" 
    "You will be presented with configuration options from the technologies '{technologyA}' and '{technologyB}' found in project '{project}'.\n" 
    "Your task is to determine whether the given configuration options actually depend on each other based on the specified type of dependency."
)

TASK_STR = PromptTemplate(
    "Given the following configuration options found in project '{project}':\n"
    "Option A: The configuration option named '{nameA}' with the value '{valueA}' of type '{typeA}' located in '{fileA}', which belongs to the technology '{technologyA}'.\n"
    "Option B: The configuration option named '{nameB}' with the value '{valueB}' of type '{typeB}' located in '{fileB}', which belongs to the technology '{technologyB}'.\n\n"
    "Validate step by step whether the given configuration options actually depend on each other due to a '{dependency_category}' dependency.\n" 
    "Respond in a JSON format as shown below:\n"
    "{{\n"
    "'rationale': string, // Provide a concise explanation of whether and why the configuration options depend on each other.\n"
    "'uncertainty': integer, // Rate your certainty of this dependency on a scale from 0 (completely uncertain) to 10 (absolutely certain).\n"
    "'isDependency': boolean // Indicate True if a configuration dependency exists, or False otherwise.\n"
    "}}"
)

