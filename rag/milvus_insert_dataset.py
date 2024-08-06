import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection, utility
)
import time

start_time  = time.time()
# Load embeddings from local file
input_file = 'squad1.1_embeddings.json'
with open(input_file, 'r') as f:
    data = json.load(f)
end_time = time.time()
print(f"Finish loading data from JSON")
print(f"Loading local data time: {end_time - start_time} seconds\n")

# Connect to Milvus
connections.connect(host='localhost', port='19530')

# Define Milvus Collection schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=255),
]
schema = CollectionSchema(fields, "SQuAD1.1 Collection")

# Create collection if it does not exist
collection_name = "squad_collection"
if not utility.has_collection(collection_name):
    collection = Collection(collection_name, schema)
else:
    collection = Collection(collection_name)
print("create \"squad_collection\" collection")

start_time  = time.time()
# Extract and store data in Milvus
contexts = [item['context'] for item in data]
embeddings = [item['embedding'] for item in data]
titles = [item['title'] for item in data]

end_time = time.time()
print(f"重新赋值时间: {end_time - start_time} seconds\n")
# for i in range(10):
#     print(titles[i])


sequential_id = list(range(len(embeddings)))
# # Insert data into Milvus
# insert_data = [
#     sequential_id,
#     contexts,  # This corresponds to the 'content' field
#     embeddings,  # This corresponds to the 'embeddings' field
#     titles
# ]



# Function to insert data in batches
def insert_in_batches(collection, sequential_id, contexts, embeddings, titles, batch_size=1000):
    total_count = len(contexts)
    for i in range(0, total_count, batch_size):
        end = min(i + batch_size, total_count)
        batch_id = sequential_id[i:end]
        batch_contexts = contexts[i:end]
        batch_embeddings = embeddings[i:end]
        batch_titles = titles[i:end]
        data_to_insert = [
            batch_id,
            batch_contexts,
            batch_embeddings,
            batch_titles
        ]
        
        insert_result = collection.insert(data_to_insert)
        print(f"Inserted batch {i//batch_size + 1}: {insert_result.insert_count} entities")
        
        # Optional: Flush after each batch
        collection.flush()


start_time  = time.time()
# Insert the data in batches
insert_in_batches(collection, sequential_id, contexts, embeddings, titles)
end_time = time.time()
print(f"Insertion time: {end_time - start_time} seconds\n")

# Flush the collection to make the data persistent and queryable
collection.flush()

# Define the index parameters for HNSW
index_params = {
    "metric_type": "IP",  # Using Inner Product for cosine similarity
    "index_type": "HNSW",  # HNSW index type
    "params": {"M": 16, "efConstruction": 200}  # HNSW-specific parameters
}
print("start building index")
start_time  = time.time()
collection.create_index("embeddings", index_params)
print("finish building index")
end_time = time.time()
print(f"Building index time: {end_time - start_time} seconds\n")

# Load the collection for search
collection.load()
print("collection loaded into memory, making it ready for operations such as indexing and searching.")

remaining_collections = utility.list_collections()
print("Current collections: ", remaining_collections)

Release collection
collection.release()