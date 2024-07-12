# general config
embedding_model = "qwen"
embedding_dimension = 3584
tool_name = "cfgnet"

# ingestion
splitting = "sentence"
chunk_size = 512
chunk_overlap = 50
extractors = []
num_websites = 5

# retrieval
top_k = 10
alpha = 1  
rerank = "colbert"
top_n = 5