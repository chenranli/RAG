import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection, utility
)
import torch

# Load embeddings from local file
input_file = 'squad_embeddings.json'
with open(input_file, 'r') as f:
    data = json.load(f)
print("finish loading data from JSON")

# Connect to Milvus
connections.connect(host='localhost', port='19530')

# Define Milvus Collection schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=768)
]
schema = CollectionSchema(fields, "SQuAD2.0 Collection")

# Create collection if it does not exist
collection_name = "squad_collection"
if not utility.has_collection(collection_name):
    collection = Collection(collection_name, schema)
else:
    collection = Collection(collection_name)

# Extract and store data in Milvus
embeddings = [item['embedding'] for item in data]
documents = [(item['context'], item['embedding']) for item in data]

# Insert data into Milvus
insert_data = [
    {"name": "content", "type": DataType.VARCHAR, "values": [doc[0] for doc in documents]},
    {"name": "embeddings", "type": DataType.FLOAT_VECTOR, "values": embeddings},
]
collection.insert(insert_data)
print("data inserted into collection")

# Flush the collection to make the data persistent and queryable
collection.flush()

# Load the collection
collection.load()
print("collection loaded into memory, making it ready for operations such as indexing and searching.")

# Index the collection
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "IP",
    "params": {"nlist": 128}
}

print("start building index")
collection.create_index("embeddings", index_params)
print("finish building index")

# Release collection
collection.release()
