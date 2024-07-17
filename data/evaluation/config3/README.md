# general config
pinecone_key = "4bc3fa0d-a789-4187-aa8f-d6b17d0ea6a3"
embedding_model = "qwen" # "qwen", "ollama"
embedding_dimension = 3584 # qwen=3584, openai=1536, ollama=...
tool_name = "cfgnet"
index_names = ["all"]
output_dir = "../data/evaluation/config3"

# ingestion
splitting = "sentence" # "sentence", "token", "semantic", "recursive"
chunk_size = 512
chunk_overlap = 50
extractors = [] # ["summary", "keyword", title]
num_websites = 3

# retrieval
top_k = 10
alpha = 1   #weight for sparse/dense retrieval, only used for hybrid query mode.
rerank = "sentence" # "sentence", "llm", "colbert"
top_n = 5