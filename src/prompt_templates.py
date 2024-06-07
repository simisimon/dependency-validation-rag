SYSTEM_PROMPT = """You are an expert in validating intra-technology and cross-technology configuration dependencies. 
You will be presented with configuration options from the technologies {} and {} in the project {}. 
Your task is to determine whether the given configuration options actually depend on each other based on the specified type of dependency."""

USER_PROMPT = """Validate the given dependency description step by step whether the given configuration options actually depend on each other due to value-equality.

Steps:
1. **Contextual Analysis**: Review the context information related to the configuration options from the provided project and technologies. Understand their purpose, existing dependencies, and constraints to set the foundation for your validation task.
2. **Dependency Review**: Examine the dependency description to determine the logic behind the specified type of  dependency between the configuration options. Consider how these configuration options are set and may interact within the specified project.
3. **Critical Evaluation**: Consider that “value-equality” may lead to false positives. Not all configurations with equal values necessarily depend on each other. Analyze whether such equality in this case indicates a true dependency between the specified configuration options .
4. **Conclusive Decision**: Based on the steps above, decide if a true dependency exists between the specified configuration options. If you are not completely not completely sure, …

Respond in a JSON format as shown below:
{{
  "rationale": string, // Provide a concise explanation of whether and why the configuration options depend on each other.
  "uncertainty": integer, // Rate your certainty of this dependency on a scale from 0 (completely uncertain) to 10 (absolutely certain).
  "isDependency": boolean // Indicate True if a dependency exists, or False otherwise.
}}

### Dependency Description
{}
"""

DEPENDENCY_PROMPT = """There is a potential dependency between two configuration options based on {} found in project {}:
- Option A:  The configuration option named {} with the value {} of type {} located in the file {} from the technology {}.
- Option B: The configuration option named {} with the value {} of type {} located in the file {}  from the technology {}.
Both configuration options may depend on each other due to a {} dependency."""