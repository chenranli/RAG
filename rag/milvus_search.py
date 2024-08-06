import numpy as np
from utils import search_vectordb, decompose_passage
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import time

# Connect to Milvus
connections.connect(host='localhost', port='19530')

# Load the collection
collection_name = "squad_collection"
collection = Collection(collection_name)

# Initialize the Sentence Transformer model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Define a function to search for a query vector
def search_vectors(query_text, collection, model):
    # Encode the query text to get the query vector
    query_embedding = model.encode([query_text])
    
    # Search in Milvus
    search_params = {
        "metric_type": "IP",  # Euclidean distance for HNSW index
        "params": {"ef": 200},  # ef is a hyperparameter for HNSW, controls the recall at the cost of speed
    }
    results = collection.search(
        data=query_embedding,
        anns_field="embeddings",
        param=search_params,
        limit=5,  # number of top results to retrieve
        expr=None,
        output_fields=["id", "content", "title"]
    )
    
    contents = ""
    # Extract and return the results
    for result in results[0]:
        print(f"Distance: {result.distance}, \nid: {result.id}\nContent: {result.entity.get('content')}")
        print(f"------------------------------------------------------------")
        print(type(result.entity.get('content')))
        contents = contents + "" + result.entity.get('content')

    print(f"contents: {contents}")

# Test the search with different queries
queries = [
    #"What is the capital of France?",
    #"Who wrote the book '1984'?",
    #"What is the largest mammal?",
    #"Which journalist criticized the Wii version for its controls?",
    "What is in front of the Notre Dame Main Building?",
]

for query in queries:
    print(f"Searching for: {query}")
    # search_vectors(query, collection, model)
    start_time = time.time()
    result_list = search_vectordb(query, collection, model, 5)
    end_time = time.time()
    print(f"---top-5 time: {end_time - start_time} seconds")
    
    # start_time = time.time()
    # result_list10 = search_vectordb(query, collection, model, 10)
    # end_time = time.time()
    # print(f"---top-10 time: {end_time - start_time} seconds")
    
    # start_time = time.time()
    # result_list1 = search_vectordb(query, collection, model, 1)
    # end_time = time.time()
    # print(f"---top-1 time: {end_time - start_time} seconds")
    
    sentence_list = []
    sentences = []
    # for result in result_list:
    #     print(f"title: {result['title']}") 
    #     print(f"context:\n{result['context']}\n")
    #
            #print(f"|{sentence.text.strip()}|")
    
