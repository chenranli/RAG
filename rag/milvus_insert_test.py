from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection, utility
)

# Connect to Milvus
connections.connect(host='localhost', port='19530')

# Define collection schema
dim = 8  # vector dimension
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
]
schema = CollectionSchema(fields, "A test collection")

# Create collection if it does not exist
collection_name = "test_collection"
if not utility.has_collection(collection_name):
    collection = Collection(collection_name, schema)
else:
    utility.drop_collection(collection_name)
    collection = Collection(collection_name, schema)

# Insert data
entities = [
    {"id": 0, "embedding": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]},
    {"id": 1, "embedding": [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]}
]
insert_result = collection.insert(entities)

# Print the result
print(f"Inserted {insert_result.insert_count} entities")

# Flush data to disk
collection.flush()

# Build an index
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128}
}
collection.create_index("embedding", index_params)

# Load data to memory
collection.load()

# Perform a search
search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
results = collection.search(
    data=[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]],  # vector to search
    anns_field="embedding",
    param=search_params,
    limit=5,
    expr=None
)

# Print results
for hits in results:
    for hit in hits:
        print(f"ID: {hit.id}, Distance: {hit.distance}")

# Release the collection from memory
collection.release()

# Disconnect from Milvus
connections.disconnect("default")
