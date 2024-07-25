# general config
pinecone_key = "dfffc43b-dd13-46f0-9a4d-b9041a8b2b29"
embedding_model = "openai" # "qwen", "ollama"
embedding_dimension = 1536 # qwen=3584, openai=1536, ollama=...
tool_name = "cfgnet"
index_names = ["all"]
output_dir = "../data/evaluation/config11"

# ingestion
splitting = "sentence" # "sentence", "token", "semantic", "recursive"
chunk_size = 512
chunk_overlap = 50
extractors = [] # ["summary", "keyword", title]
num_websites = 3

# retrieval
top_k = 10
alpha = 0.5  #weight for sparse/dense retrieval, only used for hybrid query mode.
rerank = "colbert" # "sentence", "llm", "colbert"
top_n = 5