from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from datasets import load_dataset
from utils import *
from pathlib import Path
import random
import sys


squad = load_dataset('squad')
squad_train = squad['train']

tokenizer = AutoTokenizer.from_pretrained("Soyoung97/RankT5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("Soyoung97/RankT5-base")

connections.connect("default", host="localhost", port="19530")
collection = Collection("squad_collection")
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

#Load the model and tokenizer from the local directory
local_model_dir = Path.home().joinpath('mistral_model_fp16', '7B-Instruct-v0.3')
local_model = AutoModelForCausalLM.from_pretrained(local_model_dir, torch_dtype=torch.float16)
local_tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
device = 0 if torch.cuda.is_available() else -1
chatbot = pipeline("text-generation", model=local_model, tokenizer=local_tokenizer, device=device)




random_indices = random.sample(range(1, 80001), 500)#list(range(0, 5))
results = []

for index in random_indices:
    question = squad_train[index]['question']
    answer = squad_train[index]['answers']['text']
    title = squad_train[index]['title']
    print(f"---context:\n{squad_train[index]['context']}")
    print(f"---question:\n{question}")
    print(f"---answers:\n{squad_train[index]['answers']}")
    #print(f"answers_text: {answer}")
    print(f"---title: {title}")
    
    search_result = search_vectordb(question, collection, embedding_model, 1)
    start_time = time.time()
    #generate response
    response = generate_response(search_result[0]['context'], question, search_result[0]['title'], chatbot, local_tokenizer)
    end_time = time.time()
    num_words = len(search_result[0]['context'].split())
    print(f"---response time: {end_time - start_time} seconds")
    print(f"---context word count: {num_words}")
    print(f"---Answer: {response['response']}")
    print(f"---Context token count: {response['num_tokens']}")
    accuracy = check_answer(response['response'], answer[0])
    print(f"---accuracy: {accuracy}")
    result = {
        "question_id": index,
        #"question": question,
        #"context": context,
        #"response": response,
        "response_time": end_time - start_time,
        "token_count": response['num_tokens'],
        "word_count": num_words,
        "accuracy": accuracy
    }
    results.append(result)
    
    
    #rerank
    # sentences = decompose_passage(result_list[0]['context'])
    # score_tuple_list = rerank_passages(question, title, sentences, model, tokenizer)#[(score, sentence)...]
    # ranked_sentences = sorted(score_tuple_list, key=lambda x: x[0], reverse=True)

    # for i, (score, passage) in enumerate(ranked_sentences, 1):
    #     print(f"{i}. Score: {score:.4f} - {len(passage.split())} - {passage}")




sys.exit() 

#file_path = 'lcr.txt'
# Write to file
with open(file_path, 'w') as file:
    json.dump(results, file, ensure_ascii=False, indent=4)
 
        


print(f"Results saved to {file_path}")


#print(f"{squad['train'][args.arg1]}\n")
