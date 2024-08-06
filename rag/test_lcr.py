from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from datasets import load_dataset
from utils import *
from pathlib import Path
import random
import sys
import gc


squad = load_dataset('squad')
squad_train = squad['train']

rerank_tokenizer = AutoTokenizer.from_pretrained("Soyoung97/RankT5-base")
rerank_model = AutoModelForSeq2SeqLM.from_pretrained("Soyoung97/RankT5-base")

connections.connect("default", host="localhost", port="19530")
collection = Collection("squad_collection")
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

#Load the model and tokenizer from the local directory
# local_model_dir = Path.home().joinpath('mistral_model_fp16', '7B-Instruct-v0.3')
# local_model = AutoModelForCausalLM.from_pretrained(local_model_dir, torch_dtype=torch.float16)
# local_tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
# device = 0 if torch.cuda.is_available() else -1
# chatbot = pipeline("text-generation", model=local_model, tokenizer=local_tokenizer, device=device)


# with open('baseline300.txt', 'r') as file:
#         baseline_data = json.load(file)

                # 1         1               1
#random_indices = [454, 423, 263, 478, 479, 480, 481]#random.sample(range(1, 80001), 300)#list(range(0, 5))
random_indices = random.sample(range(20001, 30000), 1)
results = []

#rerank_passages("Capital of France?", ["Paris"], ["Paris is capital of France."], rerank_model, rerank_tokenizer)
rerank_times = []
rerank_times1 = []


for index in random_indices:
    question = squad_train[index]['question']
    answer = squad_train[index]['answers']['text']
    title = squad_train[index]['title']
    #print(f"---context:\n{squad_train[index]['context']}")
    print(f"---question:\n{question}")
    print(f"---answers:\n{squad_train[index]['answers']}")
    #print(f"answers_text: {answer}")
    print(f"---title: {title}")
    
    time1 = time.time()
    passages, titles, _= search_vectordb(question, collection, embedding_model, 3)
    time2 = time.time()
    print(f"---search time: {time2 - time1} seconds\n")
    
    clear_memory()

    #dslr
    sentences_dslr = decompose_passage(passages[0])
    sentence_count = len(sentences_dslr)
    time5 = time.time()#rerank origial top-1
    relevance_scores3, sentences_dslr_rerank, _ = rerank_passages(question, [titles[0]]*sentence_count, sentences_dslr, rerank_model, rerank_tokenizer)
    #rerank_times.append(rerank_time)
    time6 = time.time()
    print(f"---senteces rerank time2: {time6 - time5}")
    sentences_dslr_rerank = [sentence for relevance_score, sentence in zip(relevance_scores3, sentences_dslr_rerank) if relevance_score >= -3.8]
    # print(f"---before reranking:\n{passages[0]}")
    # print(f"---after reranking:\n{reranked_sentences}")
    rerank_times.append((time6-time5))
    combined_sentence_dslr = ''.join(sentences_dslr_rerank)
    print(combined_sentence_dslr)
    print()


    #rerank_passages("Capital of France?", ["Paris"], ["Paris is capital of France."], rerank_model, rerank_tokenizer)
    #lcr
    time3 = time.time()
    relevance_scores1, passages, rerank_time = rerank_passages(question, titles, passages, rerank_model, rerank_tokenizer)
    time4 = time.time()
    print(f"---passages rerank time1: {time4 - time3}")
    
    max_score, max_passage, max_title = max(zip(relevance_scores1, passages, titles), key=lambda x: x[0])
    max_sentences = decompose_passage(max_passage)
    sentence_count = len(max_sentences)
    time33 = time.time()#rerank new top-1
    relevance_scores2, pas_rerank_sentences, _ = rerank_passages(question, [max_title]*sentence_count, max_sentences, rerank_model, rerank_tokenizer)
    time44 = time.time()
    print(f"---senteces rerank time1: {time44 - time33}")
    sentences_rerank = [sentence for relevance_score, sentence in zip(relevance_scores2, pas_rerank_sentences) if relevance_score >= -3.8]
    combined_sentence_rerank = ''.join(sentences_rerank)
    print(combined_sentence_rerank)
    print()

    #rerank_tokenizer, rerank_model = load_reranker_model()
    # time_test = time.time()
    # rerank_passages("Capital of France?", ["Paris"], ["Paris is capital of France."], rerank_model, rerank_tokenizer)
    # time_test1 = time.time()
    # print(f"test time: {time_test1 - time_test}")
    
    
    
    #generate response
    # del rerank_model
    # del rerank_tokenizer
    # clear_memory()

    # time9 = time.time()
    # generated_answer, num_tokens, _ = generate_response(passages[0], question, titles[0], chatbot, local_tokenizer)
    # time10 = time.time()
    
    
    # time7 = time.time()
    # generated_answer_rerank, num_tokens_rerank, _ = generate_response(combined_sentence, question, titles[0], chatbot, local_tokenizer)
    # time8 = time.time()
    
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
    result = {
        "question_id": index,
        # "response_time": (time10 - time9) * 2,
        # "response_time_rerank": (time8 - time7) * 2,
        # "token_count": num_tokens,
        # "token_count_rerank": num_tokens_rerank,
        # "word_count": num_words,
        # "word_count_rerank": num_words_rerank,
        # "accuracy": accuracy,
        # "accuracy_rerank": accuracy_rerank,
        "search_time": time2 - time1,
        "question": question,
        "rerank_pas_sen_time": time4 - time3 + time44 - time33,
        "rerank_dslr_time": time6 - time5,
        "title_pas_sen": max_title,
        "title_dslr": titles[0],
        "context_pas_sen": combined_sentence_rerank,
        "context_dslr   ": combined_sentence_dslr,
        "ground_truth_sen": find_answer_sentence(squad_train[index]['context'], squad_train[index]['answers']['answer_start'][0]),
        "ground_truth_answer": answer[0],
        "context_baseline": passages[0]
        

        #"test_time": time_test1-time_test
        #"question": question,
        #"context": context,
        #"response": response,
    }
    print(result)
    results.append(result)
    
    
    #rerank
    # sentences = decompose_passage(result_list[0]['context'])
    # score_tuple_list = rerank_passages(question, title, sentences, model, tokenizer)#[(score, sentence)...]
    # ranked_sentences = sorted(score_tuple_list, key=lambda x: x[0], reverse=True)

    # for i, (score, passage) in enumerate(ranked_sentences, 1):
    #     print(f"{i}. Score: {score:.4f} - {len(passage.split())} - {passage}")

#print(rerank_times)

sys.exit() 

file_path = 'lcr_pas_sen_?_500.txt'
# Write to file
with open(file_path, 'w') as file:
    json.dump(results, file, ensure_ascii=False, indent=4)
 
        


print(f"Results saved to {file_path}")


#print(f"{squad['train'][args.arg1]}\n")
