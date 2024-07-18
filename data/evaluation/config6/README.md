# general config
embedding_model = "qwen"
embedding_dimension = 3584
tool_name = "cfgnet"

# ingestion
splitting = "sentence"
chunk_size = 512
chunk_overlap = 50
extractors = []
num_websites = 3

# retrieval
top_k = 10
alpha = 0.6 
rerank = "colbert"
top_n = 5