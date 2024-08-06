#import en_core_web_sm
import numpy as np
#from pymilvus import connections, Collection
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection, utility
)
import json
import spacy
import torch
import time
import torch.nn as nn
import numpy as np
import re
import gc

def decompose_passage(passage):
    #nlp = en_core_web_sm.load()
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    # Process the passage
    doc = nlp(passage.strip())

    # Extract sentences
    sents = list(doc.sents)
    #print(f"type of sents: {type(sents)}")  = <class 'list'>
    sentences = [sent.text.strip() for sent in sents]
    return sentences

# def decompose_passages(passages):
#     # nlp = en_core_web_sm.load()
#     # sentence_list = []
#     # for passage in passages:
#     #     doc = nlp(passage.strip())
#     #     sentence_list.append(list(doc.sents))
#     sentences_list = []
#     nlp = spacy.blank("en")
#     # Add the sentencizer to the pipeline
#     nlp.add_pipe("sentencizer")
#     for passage in passages:
#         doc = nlp(passage.strip())
#         sents = list(doc.sents)
#         sentences = [sent.text.strip() for sent in sents]
#         sentences_list.append(sentences)
#     return sentences


def load_reranker_model(model_name = "Soyoung97/RankT5-base"):
    rerank_tokenizer = AutoTokenizer.from_pretrained("Soyoung97/RankT5-base")
    rerank_model = AutoModelForSeq2SeqLM.from_pretrained("Soyoung97/RankT5-base")
    return rerank_tokenizer, rerank_model

def connect_to_milvus(collection_name = "squad_collection", ):
    # Connect to Milvus server (assuming Milvus is running locally on default port)
    connections.connect("default", host="localhost", port="19530")

    # Define the collection name
    collection_name = "squad_collection"

# Define a function to search for a query vector
def search_vectordb(query_text, collection, model, passages_num):
    # Encode the query text to get the query vector
    query_embedding = model.encode([query_text])

    search_start_time = time.time()

    # Search in Milvus
    search_params = {
        "metric_type": "IP",  
        "params": {"ef": 200},  # ef is a hyperparameter for HNSW, controls the recall at the cost of speed
    }
    results = collection.search(
        data=query_embedding,
        anns_field="embeddings",
        param=search_params,
        limit=passages_num,  # number of top results to retrieve
        expr=None,
        output_fields=["id", "content", "title"]
    )
    #print(f"---result:\n{results}") --- data:[], cost:. !!!!!!
    search_end_time = time.time()
    #print(f"---search time: {search_end_time - search_start_time} seconds")
    #print(f"contents: {contents}")
    return ([result.entity.get('content') for result in results[0]],
        [result.entity.get('title') for result in results[0]], search_end_time - search_start_time)

#return 
def rerank_passages(question, titles, passages, model, tokenizer):
    #input_pairs = [f"Query: {sentence} Document: {passage}" for passage in passages]
    input_pairs = [f"Query: {question} Document Title: {title} Document: {passage}" for title, passage in zip(titles, passages)]
    #print(f"input_pairs:\n{input_pairs}")
    print(f"---titiles: {titles}")
    rerank_start_time = time.time()
    inputs = tokenizer(input_pairs, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, return_dict_in_generate=True, output_scores=True, max_new_tokens=1)

    rerank_end_time = time.time()
    #print(f"--- inside rerank function rerank time: {rerank_end_time - rerank_start_time} seconds")
    #scores = outputs.scores[0][:, 1]  # Assuming 1 is the token ID for "true" or high relevance

    #print(f"list length: {len(outputs.scores[0][0])}")
    print(f"list length: {len(outputs.scores[0])}")
    #print(f"outputs.scores: {outputs.scores[0]}")
    # Define a linear layer to map embeddings to a single relevance score
    # linear_layer = nn.Linear(32128, 1)
    # # Calculate the relevance score
    # relevance_score = linear_layer(outputs.scores[0])
    # print(relevance_score.item())

    # Print the first few tokens in the vocabulary
    # for i in range(10):
    #     print(f"Token {i}: {tokenizer.decode([i])}")
    # 1. Shape of the generated sequences
    #print("Shape of sequences:", outputs.sequences.shape) - Shape of sequences: torch.Size([1, 2])
    # 2. Shape of the scores
    # if hasattr(outputs, 'scores') and outputs.scores:
    #     print("Shape of scores:", [score.shape for score in outputs.scores]) - torch.Size([1, 32128])
    #print("Available attributes:", outputs.keys()) - Available attributes: odict_keys(['sequences', 'scores', 'past_key_values'])
    #print(outputs.scores[0].shape) #torch.Size([number of passages, 32128])
    relevance_scores = outputs.scores[0].mean(dim=1)
    #print(f"relevance_scores: {relevance_scores}")
    
    #ranked_passages = sorted(zip(relevance_scores, passages), key=lambda x: x[0], reverse=True)
    #print(f"type of ranked_passages is: {type(ranked_passages)}") Its type is: list, a list of (content, score) tuple
    
    return relevance_scores, passages, rerank_end_time - rerank_start_time



def check_answer(response, ground_truth):
    sentences = decompose_passage(response)
    response_words = set([word for sentence in sentences 
                    for word in re.sub(r'\.$', '', sentence.strip().lower()).split()])
    ground_truth_words = set(ground_truth.lower().split())
    missing_words = ground_truth_words - response_words
    #print(f"---response_words:\n{response_words}")
    if not missing_words:
        return True#, []
    else:
        return False#, list(missing_words)


def generate_response(context, question, title, chatbot, tokenizer, max_length=1500):
    # Generate a response
    # response = chatbot(prompt, max_length=max_length, num_return_sequences=1, temperature=0.7, top_p=0.95, do_sample=True)
    generate_start_time = time.time()
    messages = [
        {"role": "system", "content": "You are a helpful chatbot. Your task is to respond to the user’s question by looking into context."},
        {"role": "user", "content": f'''\n<s>Context:\n{title}\n{context}</s>\n\n<s>Based on this information, please answer the question: {question}</s>'''},
    ]
    messages1 = [
        {"role": "system", "content": "You are a helpful chatbot. Your task is to respond to the user’s question by looking into context."},
        {"role": "user", "content": f"\n<s>Context: {title}\n</s>\n\n<s>Based on this information, please answer the question: {question}</s>"},
    ]
    #print(f'''Real prompt:\n{messages[0]['content']}{messages[1]['content']}''')

    # Tokenize the input prompt to get the number of input tokens
    input_tokens = tokenizer.apply_chat_template(messages, return_tensors='pt')
    input_token_length = input_tokens.shape[1]
    input_tokens1 = tokenizer.apply_chat_template(messages1, return_tensors='pt')
    input_token_length1 = input_tokens1.shape[1]
    # Generate a response using the pipeline. The pipeline handles tokenization internally, so we don't need to tokenize separately
    response = chatbot(
        messages, max_length=max_length, num_return_sequences=1, truncation=True  # Enable truncation within the pipeline
        #temperature=0.7, top_p=0.95, do_sample=True
    )
    # Extract the generated text
    generated_text = response[0]['generated_text']
    # Tokenize the generated response to get the number of output tokens
    # output_tokens = local_tokenizer.apply_chat_template(generated_text, return_tensors='pt')
    # output_token_length = output_tokens.shape[1]
    # Print the lengths of input and output tokens
    #print(f"Context token length: {input_token_length - input_token_length1}")
    # print(f"original template token count: {input_tokens1.shape[1]}")
    # print(f"Output token length: {output_token_length}\n")
    generate_end_time = time.time()
    return generated_text[2]['content'], input_token_length-input_token_length1, generate_end_time - generate_start_time


def find_answer_sentence(context, answer_start):
    sentences = decompose_passage(context)
    current_position = 0
    answer_index = 0
    for i, sentence in enumerate(sentences):
        sentence_end = current_position + len(sentence)
        if current_position <= answer_start < sentence_end:
            answer_index = i
            break
        current_position = sentence_end + 1
    return sentences[answer_index]





def generate_response_test(context, question, title, chatbot, tokenizer, max_length=1500):
    # Generate a response
    # response = chatbot(prompt, max_length=max_length, num_return_sequences=1, temperature=0.7, top_p=0.95, do_sample=True)
    generate_start_time = time.time()
    messages = [
        {"role": "system", "content": "You are a helpful chatbot. Your task is to respond to the user’s question by looking into context."},
        {"role": "user", "content": f'''\n<s>Context:\n{title}\n{context}</s>\n\n<s>Based on this information, please answer the question: {question}</s>'''},
    ]
    messages2 = [
        {"role": "system", "content": "You are a helpful chatbot. Your task is to respond to the user’s question by looking into context."},
        {"role": "user", "content": f'''\n<s>Context:\n{title}\n</s>\n\n<s>Based on this information, please answer the question: {question}</s>'''},
    ]
    messages1 = [
        {"role": "system", "content": ""},
        {"role": "user", "content": f""},
    ]
    #print(f'''Real prompt:\n{messages[0]['content']}{messages[1]['content']}''')

    # Tokenize the input prompt to get the number of input tokens
    input_tokens = tokenizer.apply_chat_template(messages, return_tensors='pt')
    input_token_length = input_tokens.shape[1]
    input_tokens1 = tokenizer.apply_chat_template(messages1, return_tensors='pt')
    input_token_length1 = input_tokens1.shape[1]
    input_tokens2 = tokenizer.apply_chat_template(messages2, return_tensors='pt')
    input_token_length2 = input_tokens2.shape[1]
    print(f'''input_token_length1 = {input_token_length1}''')
    print(f"test encoder(): {len(tokenizer.encode(context))}")
    print(f"num_tokens when put in template: {input_token_length - input_token_length2}")
    for token in tokenizer.encode(context):
        print(tokenizer.decode(token))
    return
    # Generate a response using the pipeline. The pipeline handles tokenization internally, so we don't need to tokenize separately
    response = chatbot(
        messages, max_length=max_length, num_return_sequences=1, truncation=True  # Enable truncation within the pipeline
        #temperature=0.7, top_p=0.95, do_sample=True
    )
    # Extract the generated text
    generated_text = response[0]['generated_text']
    # Tokenize the generated response to get the number of output tokens
    # output_tokens = local_tokenizer.apply_chat_template(generated_text, return_tensors='pt')
    # output_token_length = output_tokens.shape[1]
    # Print the lengths of input and output tokens
    #print(f"Context token length: {input_token_length - input_token_length1}")
    # print(f"original template token count: {input_tokens1.shape[1]}")
    # print(f"Output token length: {output_token_length}\n")
    generate_end_time = time.time()
    return generated_text[2]['content'], input_token_length-input_token_length1, generate_end_time - generate_start_time







def clear_memory():
    # Clear GPU cache
    torch.cuda.empty_cache()
    # Run garbage collection to free up CPU memory
    gc.collect()
