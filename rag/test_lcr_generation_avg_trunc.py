from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from datasets import load_dataset
from utils import *
from pathlib import Path
import random
import sys
import gc


# squad = load_dataset('squad')
# squad_train = squad['train']

rerank_tokenizer = AutoTokenizer.from_pretrained("Soyoung97/RankT5-base")
rerank_model = AutoModelForSeq2SeqLM.from_pretrained("Soyoung97/RankT5-base")

connections.connect("default", host="localhost", port="19530")
collection = Collection("squad_collection")
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

#Load the model and tokenizer from the local directory
local_model_dir = Path.home().joinpath('mistral_model_fp16', '7B-Instruct-v0.3')
local_model = AutoModelForCausalLM.from_pretrained(local_model_dir, torch_dtype=torch.float16)
local_tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
device = 0 if torch.cuda.is_available() else -1
chatbot = pipeline("text-generation", model=local_model, tokenizer=local_tokenizer, device=device)


# with open('lcr_pas_sen_0_10000_500.txt', 'r') as file:
#         local_data = json.load(file)
with open('lcr_pas_sen_0_10000_500.txt', 'r') as file:
        local_data = json.load(file)

#random_indices = [454, 423, 263, 478, 479, 480, 481]#random.sample(range(1, 80001), 300)#list(range(0, 5))
#random_indices = random.sample(range(1, 80001), 300)
results = []

#rerank_passages("Capital of France?", ["Paris"], ["Paris is capital of France."], rerank_model, rerank_tokenizer)
rerank_times = []
rerank_times1 = []

#print(list(local_data[0].keys()))

question_ids = [entry['question_id'] for entry in local_data]
context_pas_sen_list = [entry['context_pas_sen'] for entry in local_data]
context_dslr_list = [entry['context_dslr   '] for entry in local_data]
ground_truth_answer_list = [entry['ground_truth_answer'] for entry in local_data]
question_list = [entry['question'] for entry in local_data]
title_pas_sen_list = [entry['title_pas_sen'] for entry in local_data]
title_dslr_list = [entry['title_dslr'] for entry in local_data]
context_baseline_list = [entry['context_baseline'] for entry in local_data]
                             
count = 0
token_limit = 64
for index in question_ids:
    # time1 = time.time()
    # passages, titles, _= search_vectordb(question_list[count], collection, embedding_model, 1)
    # time2 = time.time()
    # print(f"---search time: {time2 - time1} seconds\n")

    clear_memory()

    question = question_list[count]
    title = title_dslr_list[count]
    #concatenate most relevant sentences until meet tokens count limit
    sentences = decompose_passage(context_baseline_list[count])
    relevance_scores, rerank_sentences, _ = rerank_passages(question, [title]*len(sentences), sentences, rerank_model, rerank_tokenizer)
    score_sen_pairs = sorted(list(zip(relevance_scores, rerank_sentences)), key=lambda x: x[0], reverse=True)
    sen_tokens_nums = [len(local_tokenizer.encode(sen)) for score, sen in score_sen_pairs]
    print(sen_tokens_nums)
    combined_sentences_rerank = []
    current_token_num = 0
    for sen_token_num, score_sen_pair in zip(sen_tokens_nums, score_sen_pairs):
        if current_token_num + sen_token_num <= token_limit:
            combined_sentences_rerank.append(score_sen_pair[1])
            current_token_num += sen_token_num
        else:
            break
    generated_answer_truncat_rank = generate_response(combined_sentences_rerank, question, title, chatbot, local_tokenizer)
    #print(f"---question:\n{question}")
    print(f"---combined_sentences:\n{combined_sentences_rerank}")
    #print(f"---ground_truth_answer_list:\n{ground_truth_answer_list[count]}")
    #print(reponse)

    
    sen_tokens_nums1 = [len(local_tokenizer.encode(sen)) for sen in sentences]
    combined_sentences_order = []
    current_token_num1 = 0
    for sen_token_num, sentence in zip(sen_tokens_nums, sentences):
        if current_token_num1 + sen_token_num <= token_limit:
            combined_sentences_order.append(sentence)
            current_token_num1 += sen_token_num
        else:
            break
    generated_answer_truncat_order = generate_response(combined_sentences_order, question, title, chatbot, local_tokenizer)
    print(f"---combined_sentences:\n{combined_sentences_rerank}")
    
    if count == 1:
        break
    # time7 = time.time()
    # generated_answer_pas_sen, num_tokens_pas_sen, _ = generate_response(context_pas_sen_list[count], question_list[count], title_pas_sen_list[count], chatbot, local_tokenizer)
    # time8 = time.time()
    
    # time5 = time.time()
    # generated_answer_dslr, num_tokens_dslr, _ = generate_response(context_dslr_list[count], question_list[count], title_dslr_list[count], chatbot, local_tokenizer)
    # time6 = time.time()

    # time3 = time.time()
    # generated_answer_baseline, num_tokens_baseline, _ = generate_response(context_baseline_list[count], question_list[count], title_dslr_list[count], chatbot, local_tokenizer)
    # time4 = time.time()

    # num_words = len(passages[0].split())
    # num_words_rerank = len(combined_sentence.split())
    # print(f"---response_time: {(time10 - time9)*2} seconds")
    # print(f"---response_time_rerank: {(time8 - time7)*2} seconds")
    # print(f"---context word count: {num_words}")
    # print(f"---context word count rerank: {num_words_rerank}")
    # print(f"---rerank time: {time6 - time5}")
    # #print(f"---Answer: {generated_answer}")
    # accuracy = check_answer(generated_answer, answer[0])
    # accuracy_rerank = check_answer(generated_answer_rerank, answer[0])
    # print(f"---accuracy: {accuracy}")
    # print(f"---accuracy_rerank: {accuracy_rerank}")

    # accuracy_baseline = check_answer(generated_answer_baseline, ground_truth_answer_list[count])
    # accuracy_dslr = check_answer(generated_answer_dslr, ground_truth_answer_list[count])
    # accuracy_pas_sen = check_answer(generated_answer_pas_sen, ground_truth_answer_list[count])
    result = {
        "question_id": index,
        # "response_time": (time10 - time9) * 2,
        # "response_time_rerank": (time8 - time7) * 2,
        # "token_count": num_tokens,
        # "token_count_rerank": num_tokens_rerank,
        # "word_count": num_words,
        # "word_count_rerank": num_words_rerank,
        # "accuracy": accuracy_baseline,
        # "accuracy_dslr": accuracy_dslr,
        # "accuracy_pas_sen": accuracy_pas_sen,
        #"search_time": time2 - time1,
        # "num_tokens_baseline": num_tokens_baseline,
        # "num_tokens_dslr": num_tokens_dslr,
        # "num_tokens_pas_sen": num_tokens_pas_sen,
        #"question": question,
        # "gen_baseline": (time4 - time3)*3,
        # "gen_dslr_time": (time6 - time5)*3,
        # "gen_pas_sen_time": (time8 - time7)*3,
        #"title_pas_sen": max_title,
        #"title_dslr": titles[0],
        #"context_pas_sen": combined_sentence_rerank,
        #"context_dslr   ": combined_sentence_dslr,
        #"gound_truth_sen": find_answer_sentence(squad_train[index]['context'], squad_train[index]['answers']['answer_start'][0]),
        #"gound_truth_anser": answer[0]
    }
    print(f'''{result}\n''')
    results.append(result)
    count += 1
    # if count == 10:
    #     break
    
    #rerank
    # sentences = decompose_passage(result_list[0]['context'])
    # score_tuple_list = rerank_passages(question, title, sentences, model, tokenizer)#[(score, sentence)...]
    # ranked_sentences = sorted(score_tuple_list, key=lambda x: x[0], reverse=True)

    # for i, (score, passage) in enumerate(ranked_sentences, 1):
    #     print(f"{i}. Score: {score:.4f} - {len(passage.split())} - {passage}")

#print(rerank_times)

sys.exit() 

file_path = 'lcr_gen_?_500.txt'
# Write to file
with open(file_path, 'w') as file:
    json.dump(results, file, ensure_ascii=False, indent=4)
 

print(f"Results saved to {file_path}")


#print(f"{squad['train'][args.arg1]}\n")
