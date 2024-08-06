from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection, utility
)

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

remaining_collections = utility.list_collections()
print("Before: ", remaining_collections)


collection_name = "squad_collection"
# Drop the collection
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

remaining_collections = utility.list_collections()
print("After: ", remaining_collections)

# Disconnect from Milvus
connections.disconnect("default")
