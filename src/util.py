def is_index_empty(index) -> bool:
    vector_count = index.describe_index_stats()["total_vector_count"]
    if vector_count == 0:
        return True
    else: 
        return False