import numpy as np
import torch
from utils import search_vectordb, decompose_passage, check_answer
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from pathlib import Path
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import random

device = 0 if torch.cuda.is_available() else -1
# Connect to Milvus
connections.connect(host='localhost', port='19530')
# Load the collection
collection_name = "squad_collection"
collection = Collection(collection_name)
# Initialize the Sentence Transformer model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
#Load the model and tokenizer from the local directory
local_model_dir = Path.home().joinpath('mistral_model_fp16', '7B-Instruct-v0.3')
local_model = AutoModelForCausalLM.from_pretrained(local_model_dir, torch_dtype=torch.float16)
local_tokenizer = AutoTokenizer.from_pretrained(local_model_dir)

# Load SQuAD1.1
squad_train = load_dataset('squad')['train']
print(f"{squad_train[1]}\n")

index = [1]

random_indices = random.sample(range(1, 50001), 100)
result_list = []

for index in random_indices:
    question = squad_train[index]['question']
    answer = squad_train[index]['answers']['text']
    # for index, passage in enumerate(passage_list):
    #     print(f"Passage{index}:\n{passage}\n")

    title = squad_train[index]['title']
    #context =  squad_train[index]['context']
    passage_list = search_vectordb(question, collection, model, 1)
    context = passage_list[0][1]
    #context = truncate_passage(context)
    
    messages = [
        {"role": "system", "content": "You are a helpful chatbot. Your task is to respond to the user’s question by looking into context."},
        {"role": "user", "content": f"\n<s>Context: {context}</s>\n\n<s>Based on this information, please answer the question: {question}</s>"},
    ]

    chatbot = pipeline("text-generation", model=local_model, tokenizer=local_tokenizer, device=device)
    response = generate_response(messages)
    #print(f"Prompt: {messages}\n")
    #print(f"Response:{response}\n")
    print(f"Chatbot answer:\n{response[2]['content']}")
    print(f"Ground truth answer:\n{answer}")
    correctness, missing_word = check_answer(response[2]['content'], answer[0])
    print(f"accuracy: {correctness}, missing word: {missing_word}")





# Write the random numbers to the file
with open(file_name, "w") as file:
    for number in random_numbers:
        file.write(f"{number}\n")

print(f"100 random numbers have been written to {file_name}")



    




def generate_response(prompt, max_length=1000):
    # Generate a response
    # response = chatbot(prompt, max_length=max_length, num_return_sequences=1, temperature=0.7, top_p=0.95, do_sample=True)
    input_string = ""
    for item in prompt:
        if item["role"] == "system":
            input_string += f"<s>{item['content']}</s>\n"
        elif item["role"] == "user":
            input_string += f"{item['content']}\n"

    #print(f"Real prompt: {input_string}")
    # Tokenize the input prompt to get the number of input tokens
    input_tokens = local_tokenizer.apply_chat_template(prompt, return_tensors='pt')
    input_token_length = input_tokens.shape[1]

    # Generate a response using the pipeline
    # The pipeline handles tokenization internally, so we don't need to tokenize separately
    response = chatbot(
        prompt, max_length=max_length, num_return_sequences=1, truncation=True  # Enable truncation within the pipeline
        #temperature=0.7, #top_p=0.95, #do_sample=True,
    )
    # Extract the generated text
    generated_text = response[0]['generated_text']
    # Tokenize the generated response to get the number of output tokens
    # output_tokens = local_tokenizer.apply_chat_template(generated_text, return_tensors='pt')
    # output_token_length = output_tokens.shape[1]
    print(f"Input token length: {input_token_length}")
    # print(f"Output token length: {output_token_length}\n")
    return generated_text

