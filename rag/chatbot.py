from pymilvus import Collection, connections, utility
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch

# Load the model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Initialize the Sentence Transformer model
sentence_transformer_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Step 1: Retrieve documents from Milvus
def retrieve_documents(query, collection, model):
    # Create embeddings for the query
    query_embedding = model.encode([query])
    
    # Search in Milvus
    search_params = {
        "metric_type": "IP",  # Inner Product (for cosine similarity)
        "params": {"nprobe": 10},
    }
    results = collection.search(
        data=query_embedding,
        anns_field="embeddings",
        param=search_params,
        limit=5,  # number of top results to retrieve
        expr=None,
        output_fields=["content"]
    )
    
    # Extract and return documents
    documents = [result.entity.get("content") for result in results[0]]
    return documents

# Step 2: Modify retrieved documents
def modify_documents(documents):
    # Example modification: Adding a custom header to each document
    modified_documents = [f"Custom Header: {doc}" for doc in documents]
    return modified_documents

# Step 3: Setup the TinyLlama generator
class TinyLlamaGenerator:
    def __init__(self, model_name):
        self.model = torch.hub.load('mit-han-lab/tiny-llama', model_name)

    def generate_response(self, prompt):
        return self.model.generate(prompt)

# Step 4: Generate response using TinyLlama with formatted prompt
def generate_response(query, retriever, generator):
    documents = retriever(query)
    modified_documents = modify_documents(documents)
    
    # Create a combined prompt from modified documents
    context = " ".join(modified_documents)
    formatted_prompt = f"""
        <s>
        You are a helpful chatbot. Your task is to respond to the user’s question by looking into context.
        </s>
        <s>
        Context: {context}
        </s>
        <s>
        Based on the above context, please answer the question: {query}
        </s>
        """

    response = generator.generate_response(formatted_prompt)
    return response

# Putting it all together
query = "What is the capital of France?"
retriever = lambda q: retrieve_documents(q, collection, model)
generator = TinyLlamaGenerator('TinyLlama-1.1B-Chat-v1.0')

response = generate_response(query, retriever, generator)
print(response)
