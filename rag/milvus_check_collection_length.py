from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection, utility
)
import json

# Connect to Milvus server (assuming Milvus is running locally on default port)
connections.connect("default", host="localhost", port="19530")

# Define the collection name
collection_name = "squad_collection"

# Load the collection
if utility.has_collection(collection_name):
    collection = Collection(name=collection_name)

    # Get the number of entities in the collection
    num_entities = collection.num_entities

    print(f"The collection '{collection_name}' contains {num_entities} entities.")
else:
    print(f"No such collection.")


