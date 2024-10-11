## Paper
PDF: will be linked later

## ABSTRACT</h3>

Retrieval-augmented generation (RAG) is an umbrella of different components, design decisions, and domain-specific adaptations to enhance the capabilities of large language models and counter their limitations regarding hallucination and outdated and missing knowledge. 
Since it is unclear which design decisions lead to a satisfactory performance, developing RAG systems is often experimental and needs to follow a systematic and sound methodology to gain sound and reliable results. However, there is currently no generally accepted methodology for RAG evaluation despite a growing interest in this technology. 

In this paper, we propose a first blueprint of a methodology for a
sound and reliable evaluation of RAG systems and demonstrate its applicability on a real-world software engineering research task: the validation of configuration dependencies across software
technologies.

In summary, we make two novel contributions: (i) A novel, reusable methodological design for evaluating RAG systems, including a demonstration that represents a guideline, and (ii) a RAG system, which has been developed following this methodology, that achieves the highest accuracy in the field of dependency validation. For the blueprint's demonstration, the key insights are the crucial role of choosing appropriate baselines and metrics, the necessity for systematic RAG refinements derived from qualitative failure analysis, as well as the reporting practices of key design decision to foster replication and evaluation.

## Project Structure

- `/data`: contains data of subject systems, ingested data, and validation results 
- `/evaluation`: contains script for evaluation
- `/src`: contains implementation the RAG system

## Supplementary Material

<details>
  <summary><h3>RQ1</h3></summary>

  We present the different RAG variants and their configuration used in our study.

<table>
  <thead>
    <tr>
      <th>ID</th>
      <th>Embedding Model</th>
      <th>Embedding Dimension</th>
      <th>Reranking</th>
      <th>Top N</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>text-embed-ada-002</td>
      <td style="text-align: right;">1536</td>
      <td>Colbert</td>
      <td style="text-align: right;">5</td>
    </tr>
    <tr>
      <td>2</td>
      <td>gte-Qwen2-7B-instruct</td>
      <td style="text-align: right;">3584</td>
      <td>Colbert</td>
      <td style="text-align: right;">5</td>
    </tr>
    <tr>
      <td>3</td>
      <td>gte-Qwen2-7B-instruct</td>
      <td style="text-align: right;">3584</td>
      <td>Sentence Transformer</td>
      <td style="text-align: right;">5</td>
    </tr>
    <tr>
      <td>4</td>
      <td>gte-Qwen2-7B-instruct</td>
      <td style="text-align: right;">3584</td>
      <td>Colbert</td>
      <td style="text-align: right;">3</td>
    </tr>
  </tbody>
</table>


</details>


<details>
  <summary><h3>RQ2</h3></summary>

We present the failure categories along with a brief description, the involved technologies, and the actionable that can be taken from them to reduce the number of failures in these categories.

<table>
  <thead>
    <tr>
      <th>Failure Cat.</th>
      <th>Description</th>
      <th>Technologies</th>
      <th>Actionable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Inheritance and Overrides</td>
      <td>Maven introduces project inheritance, allowing modules to inherit configuration from a parent module, such as dependencies, plugins, properties, and build settings. This means, for instance, while the groupID of a project generally gets inherited and is not a dependency if set explicitly, this is not true if a module is depending on another one. Meaning that in these cases the groupID has to be set explicitly and be the same.</td>
      <td>Maven</td>
      <td>Provide project-specific information on project structure and module organization</td>
    </tr>
    <tr>
      <td>Configuration Consistency</td>
      <td>Often values are the same across configuration files to ensure consistency. In this failure category, LLMs confuse equal values for the sake of consistency with dependencies.</td>
      <td>Docker-Compose, Maven, Node.js, Spring Boot</td>
      <td>Specialize prompt to distinguish consistency and dependency</td>
    </tr>
    <tr>
      <td>Resource Sharing</td>
      <td>Sometimes resources, such as databases or services can be shared across modules or used exclusively by a single module. Without additional project-specific information on how resources belong to modules, LLMs struggle to identify these dependencies.</td>
      <td>Docker-Compose, Spring</td>
      <td>Provide project-specific information on available resources</td>
    </tr>
    <tr>
      <td>Port Mapping</td>
      <td>The ports of a service (e.g., Web server) are typically defined in several configuration files of different technologies, such as <code>application.yml</code>, <code>Dockerfile</code>, and <code>Dockerfile</code>. However, not all port mappings have to be equal (e.g., a container and host port in docker compose).</td>
      <td>Docker, Docker-Compose, Spring</td>
      <td>Provide examples for port mapping dependencies and non-dependencies</td>
    </tr>
    <tr>
      <td>Ambiguous Options Names</td>
      <td>Software projects often use ambiguous naming schemes for configuration options and their values. These ambiguities result from generic and commonly used names (e.g., project name) that may not cause configuration errors if not consistent but can easily lead to misinterpretation by LLMs.</td>
      <td>Docker-Compose, Maven, Spring</td>
      <td>Specialize prompt to create awareness of naming conventions</td>
    </tr>
    <tr>
      <td>Context (Availability, Retrieval, and Utilization)</td>
      <td>Failures in this category are either because relevant information is missing (e.g., not in the vector database or generally not available to vanilla LLMs), available in the database but not retrieved, or given to the LLM but not utilized to draw the right conclusion.</td>
      <td>Docker-Compose, Maven</td>
      <td>Add context, improve sources, or improve retrieval and prompting</td>
    </tr>
    <tr>
      <td>Independent Technologies and Services</td>
      <td>In some cases (e.g., in containerized projects), different components are isolated by design. Thus, in these cases, the configuration options between these components are independent if not explicitly specified.</td>
      <td>Docker, Docker-Compose</td>
      <td>Provide examples of dependent and independent cases</td>
    </tr>
    <tr>
      <td>Others</td>
      <td>This category contains all cases in which the LLMs fail to classify the dependencies correctly that cannot be matched to any other category and share no common structure.</td>
      <td>Docker, Docker-Compose, Spring, Maven, Node.js, TSconfig</td>
      <td>Provide similar examples if possible</td>
    </tr>
  </tbody>
</table>


</details>

<details>
  <summary><h2>Running Ingestion, Retrieval, and Generation Pipeline</h2></summary>

  The RAG system consists of three pipelines that have to be executed one after the other, inluding the ingestion, retrieval, and generation pipeline. Before you run the retrieval and generation pipeline, you must first set up the vector database by running the ingestion pipeline. You can then run the retrieval pipeline to retrieve the context and afterwards the generation pipeline to generate validation responses.

  A ``.env`` file in the root directory containing the API token for OpenAI, Pinecone, and GitHub is required to run the different pipelines.

  ```
  OPENAI_KEY=<your-openai-key>
  PINECONE_API_KEY=<your-pinecone-key>
  GITHUB_TOKEN=<your-github-key>   
  ```


<details>
   <summary><h3>Run Ingestion Pipeline</h3></summary>

  For running the ingestion pipeline, there are different parameters to be adjusted in ``ingestion_config.toml``:
  - `embedding_model` defines the embedding model, e.g., qwen or openai.
  - `embedding_dimension` defines the dimension of the embedding model, e.g., 3584 for qwen or 1536 for openai.
  - `splitting` defines the splitting algorithm, e.g., sentence.
  - `urls` defines the urls that should be scraped and index into the vector database.
  - `github` defines a list of github repositories from which content should scraped and index into the vector database.
  - `data` defined a data directory of pre-processed text files that should be scraped and index into the vector database. 
  
  To run the ingestion pipeline, execute the jupyter notebooke `src/ingestion_pipeline.ipynb`.
</details>

<details>
  <summary><h3>Run Retrieval Pipeline</h3></summary>

  As soon as the vector database has been set up and filled with context information, the retrieval pipeline can be executed.
  For running the retrieval pipeline, there are different parameters to be adjusted in ``retrieval_config.toml``:
  - `embedding_model` defines the embedding model, e.g., qwen or openai.
  - `embedding_dimension` defines the dimension of the embedding model, e.g., 3584 for qwen or 1536 for openai.
  - `index_name` defines the index from which data should be retrieved, the index `all` retrieves context from all existing indices in the vector database.
  - `data_file` defines the data file containing the dependencies for which additional context should be retrieved.
  - `outfile` defines the output file (JSON) to store the dependencies with the retrieved context for dependency validation.
  - `splitting` defines the splitting algorithm, e.g., sentence..
  - `num_websites` defines the number of website to query when retrieving dynamic context for dependency validation, e.g., 3.
  - `top_k` defines the number of context chunks to retrieve.
  - `alpha ` defines the weight for sparse/dense retrieval, set to 0.5 for hybrid search.
  - `rerank` defines the re-ranking algorithm, e.g., colbert or sentence.
  - `top_n` defines the final number of context chunks that are sent to the LLM, e.g., 3 or 5.
  
  To run the retrieval pipeline, execute the Python script `src/retrieval_pipeline.py`.
</details>

<details>
  <summary><h3>Run Generation Pipeline</h3></summary>

  As soon as you obtained retrieved context from the retrieval pipeline, the generation pipeline can be executed.  
  For running the generation pipeline, there are different parameters to be adjusted in ``generation_config.toml``:
  - `data_file` defines the data file (JSON) containing the dependencies and the retrieved context for dependency validation.
  - `output_file` defines the output file (JSON) to store the validation responses.
  - `with_rag` should be `true` to run the validation with RAG, else `false`.
  - `with_refinements` should be `true` to run generation with refinements, by default `false`.
  - `model_name` defines the name of LLM used for dependency validation.
  - `temperature` defines the temperature of the LLM. Lower temperature values result into more deterministic results. Is set to 0.0.
  
  To run the generation pipeline, execute the Python script `src/generation_pipeline.py`
  </details>
</details>
