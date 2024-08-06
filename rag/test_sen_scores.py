from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from utils import *
import random
import sys

squad = load_dataset('squad')
squad_train = squad['train']
tokenizer = AutoTokenizer.from_pretrained("Soyoung97/RankT5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("Soyoung97/RankT5-base")

random_indices = [75919]#list(range(0, 100))#random.sample(range(1, 50001), 100)
result_list = []
score_list = []
for index in random_indices:
    
    question = squad_train[index]['question']
    answer = squad_train[index]['answers']['text']
    title = squad_train[index]['title']
    print(f"context: {squad_train[index]['context']}")
    print(f"question: {question}")
    print(f"answers: {squad_train[index]['answers']}")
    print(f"answers_text: {answer}")
    print(f"title: {title}")
    answer_start = squad_train[index]['answers']['answer_start'][0]
    #
    sentences = decompose_passage(squad_train[index]['context'])
    current_position = 0
    answer_index = 0
    for i, sentence in enumerate(sentences):
        sentence_end = current_position + len(sentence)
        if current_position <= answer_start < sentence_end:
            answer_index = i
            break
        current_position = sentence_end + 1
    #
    #modified_sentences = [title + " " + s for s in sentences]
    titles = [title] * len(sentences)
    start_time  = time.time()
    score_tuple_list = rerank_passages(question, titles, sentences, model, tokenizer)#[(score, sentence)...]
    end_time = time.time()
    print(f"reranking time: {end_time - start_time} seconds")
    ranked_sentences = sorted(score_tuple_list, key=lambda x: x[0], reverse=True)
    for i, (score, passage) in enumerate(ranked_sentences, 1):
        print(f"{i}. Score: {score:.4f} - {len(passage.split())} - {passage}")

    # answer_score = score_tuple_list[answer_index][0]
    # answer_sentence = score_tuple_list[answer_index][1]
    print(f"answer_score: {score_tuple_list[answer_index][0]}")
    print(f"answer_sentence: \n{score_tuple_list[answer_index][1]}\n")
    
    #score_list.append(answer_score)
    # for i, (score, sentence) in enumerate(ranked_sentences, 1):
    #     print(f"{i}. Score: {score:.4f} - {sentence}")

sys.exit() 

file_path = 'scores.txt'
# Write to file
with open(file_path, 'w') as file:
    for score in score_list:
        file.write(f"{score}\n")

print(f"List of integers saved to {file_path}")


#print(f"{squad['train'][args.arg1]}\n")
