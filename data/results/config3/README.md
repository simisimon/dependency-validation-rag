# general config
pinecone_key = "4bc3fa0d-a789-4187-aa8f-d6b17d0ea6a3"
embedding_model = "qwen"
embedding_dimension = 3584
tool_name = "cfgnet"
index_names = ["all"]
output_dir = "../data/evaluation/config13"

# ingestion
splitting = "sentence" 
chunk_size = 512
chunk_overlap = 50
extractors = [] 
num_websites = 3

# retrieval
top_k = 10
alpha = 0.5  
rerank = "sentence" 
top_n = 5