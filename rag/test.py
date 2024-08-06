from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from pathlib import Path
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import json
from utils import *
import sys
# sentences = ["This is an example sentence", "Each sentence is converted"]

# model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# embeddings = model.encode(sentences)
# print(embeddings)
# print("embedding length: ", len(embeddings[0]))
# print(f"embedding type: {type(embeddings[0])}")

# # Load SQuAD2.0
# squad_v2 = load_dataset('squad_v2')

# # Print the first example from the validation set of SQuAD2.0
# print(squad_v2['train'][0])
# print(f"type of element squad_v2['train'][0]: {type(squad_v2['train'][0])}")
# print(f"type of context:{type(squad_v2['train'][0]['context'])}")

# print("\n")
# # Print the first example from the training set
# #print(squad_v2['train'][0])
# # print(40099)
# # print(squad_v2['train'][40099])
# # print(f"-------------------------------------------------------")
# # print(40100)
# # print(squad_v2['train'][40100])
# # print(f"-------------------------------------------------------")
# # print(40101)
# # print(squad_v2['train'][40101])
# # print(f"-------------------------------------------------------")
# # print(40102)
# # print(squad_v2['train'][40102])
# # print(f"-------------------------------------------------------")


# print(len(squad_v2['train']))

# print(Path.home())


context = f'''Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.'''
answer_start = [188]
answers = {'text': ['a copper statue of Christ'], 'answer_start': [188]}
answer_text = context[answer_start[0]:]#answer_start[0] + len(answers['text'][0]) + 10]
print(answer_text)

with open('baseline300.txt', 'r') as file:
        baseline_data = json.load(file)

accuracy = sum(1 for entry in baseline_data if entry.get('accuracy') == True)
print(accuracy)

local_model_dir = Path.home().joinpath('mistral_model', '7B-Instruct-v0.3')
print(local_model_dir)

context = '''In the 1940s, new interpretations of John's reign began to emerge, based on research into the record evidence of his reign, such as pipe rolls, charters, court documents and similar primary records. Notably, an essay by Vivian Galbraith in 1945 proposed a "new approach" to understanding the ruler. The use of recorded evidence was combined with an increased scepticism about two of the most colourful chroniclers of John's reign, Roger of Wendover and Matthew Paris. In many cases the detail provided by these chroniclers, both writing after John's death, was challenged by modern historians. Interpretations of Magna Carta and the role of the rebel barons in 1215 have been significantly revised: although the charter's symbolic, constitutional value for later generations is unquestionable, in the context of John's reign most historians now consider it a failed peace agreement between "partisan" factions. There has been increasing debate about the nature of John's Irish policies. Specialists in Irish medieval history, such as Sean Duffy, have challenged the conventional narrative established by Lewis Warren, suggesting that Ireland was less stable by 1216 than was previously supposed.'''


print(find_answer_sentence(context, 7))

#------------------------------------
with open('lcr_pas_sen_0_10000_500.txt', 'r') as file:
        local_data = json.load(file)
runtime = {
        "rerank_pas_sen_time": [d["rerank_pas_sen_time"] for d in local_data if d["rerank_pas_sen_time"] < 1],
        "rerank_dslr_time": [d["rerank_dslr_time"] for d in local_data if d["rerank_dslr_time"] < 1]
    }
print(sum(runtime['rerank_pas_sen_time'])/len(runtime['rerank_pas_sen_time']))
print(sum(runtime['rerank_dslr_time'])/len(runtime['rerank_dslr_time']))
print()
#------------------------------------
with open('lcr_pas_sen_10000_20000_500.txt', 'r') as file:
        local_data = json.load(file)
runtime = {
        "rerank_pas_sen_time": [d["rerank_pas_sen_time"] for d in local_data if d["rerank_pas_sen_time"] < 1],
        "rerank_dslr_time": [d["rerank_dslr_time"] for d in local_data if d["rerank_dslr_time"] < 1]
    }
print(sum(runtime['rerank_pas_sen_time'])/len(runtime['rerank_pas_sen_time']))
print(sum(runtime['rerank_dslr_time'])/len(runtime['rerank_dslr_time']))
print()
#------------------------------------
with open('lcr_pas_sen_30000_40000_500.txt', 'r') as file:
        local_data = json.load(file)
runtime = {
        "rerank_pas_sen_time": [d["rerank_pas_sen_time"] for d in local_data if d["rerank_pas_sen_time"] < 1],
        "rerank_dslr_time": [d["rerank_dslr_time"] for d in local_data if d["rerank_dslr_time"] < 1]
    }
print(sum(runtime['rerank_pas_sen_time'])/len(runtime['rerank_pas_sen_time']))
print(sum(runtime['rerank_dslr_time'])/len(runtime['rerank_dslr_time']))
print()
#------------------------------------
with open('lcr_pas_sen_50000_60000_500.txt', 'r') as file:
        local_data = json.load(file)
runtime = {
        "rerank_pas_sen_time": [d["rerank_pas_sen_time"] for d in local_data if d["rerank_pas_sen_time"] < 1],
        "rerank_dslr_time": [d["rerank_dslr_time"] for d in local_data if d["rerank_dslr_time"] < 1]
    }
print(sum(runtime['rerank_pas_sen_time'])/len(runtime['rerank_pas_sen_time']))
print(sum(runtime['rerank_dslr_time'])/len(runtime['rerank_dslr_time']))
print()
#------------------------------------
#------------------------------------
#------------------------------------
with open('lcr_gen_0_10000_500.txt', 'r') as file:
        local_data = json.load(file)
accuracy = {
        "accuracy": sum(d["accuracy"] for d in local_data),
        "accuracy_dslr": sum(d["accuracy_dslr"] for d in local_data),
        "accuracy_pas_sen": sum(d["accuracy_pas_sen"] for d in local_data)
    }

print(accuracy)
time1 = [a["gen_baseline"] for a in local_data]
time2 = [a["gen_dslr_time"] for a in local_data]
time3 = [a["gen_pas_sen_time"] for a in local_data]
print(sum(time1)/len(time1))
print(sum(time2)/len(time2))
print(sum(time3)/len(time3))
num_tokens = {
        "num_tokens_baseline": sum(d["num_tokens_baseline"] for d in local_data)/len(time1),
        "num_tokens_dslr": sum(d["num_tokens_dslr"] for d in local_data)/len(time1),
        "num_tokens_pas_sen": sum(d["num_tokens_pas_sen"] for d in local_data)/len(time1)
    }
print(num_tokens)
print()
#------------------------------------
with open('lcr_gen_10000_20000_500.txt', 'r') as file:
        local_data = json.load(file)
accuracy = {
        "accuracy": sum(d["accuracy"] for d in local_data),
        "accuracy_dslr": sum(d["accuracy_dslr"] for d in local_data),
        "accuracy_pas_sen": sum(d["accuracy_pas_sen"] for d in local_data)
    }

print(accuracy)
time1 = [a["gen_baseline"] for a in local_data]
time2 = [a["gen_dslr_time"] for a in local_data]
time3 = [a["gen_pas_sen_time"] for a in local_data]
print(sum(time1)/len(time1))
print(sum(time2)/len(time2))
print(sum(time3)/len(time3))
num_tokens = {
        "num_tokens_baseline": sum(d["num_tokens_baseline"] for d in local_data)/len(time1),
        "num_tokens_dslr": sum(d["num_tokens_dslr"] for d in local_data)/len(time1),
        "num_tokens_pas_sen": sum(d["num_tokens_pas_sen"] for d in local_data)/len(time1)
    }
print(num_tokens)
print()
#------------------------------------
with open('lcr_gen_30000_40000_500.txt', 'r') as file:
        local_data = json.load(file)
accuracy = {
        "accuracy": sum(d["accuracy"] for d in local_data),
        "accuracy_dslr": sum(d["accuracy_dslr"] for d in local_data),
        "accuracy_pas_sen": sum(d["accuracy_pas_sen"] for d in local_data)
    }

print(accuracy)
time1 = [a["gen_baseline"] for a in local_data]
time2 = [a["gen_dslr_time"] for a in local_data]
time3 = [a["gen_pas_sen_time"] for a in local_data]
print(sum(time1)/len(time1))
print(sum(time2)/len(time2))
print(sum(time3)/len(time3))
num_tokens = {
        "num_tokens_baseline": sum(d["num_tokens_baseline"] for d in local_data)/len(time1),
        "num_tokens_dslr": sum(d["num_tokens_dslr"] for d in local_data)/len(time1),
        "num_tokens_pas_sen": sum(d["num_tokens_pas_sen"] for d in local_data)/len(time1)
    }
print(num_tokens)
print()
#------------------------------------
with open('lcr_gen_50000_60000_500.txt', 'r') as file:
        local_data = json.load(file)
accuracy = {
        "accuracy": sum(d["accuracy"] for d in local_data),
        "accuracy_dslr": sum(d["accuracy_dslr"] for d in local_data),
        "accuracy_pas_sen": sum(d["accuracy_pas_sen"] for d in local_data)
    }

print(accuracy)
time1 = [a["gen_baseline"] for a in local_data]
time2 = [a["gen_dslr_time"] for a in local_data]
time3 = [a["gen_pas_sen_time"] for a in local_data]
print(sum(time1)/len(time1))
print(sum(time2)/len(time2))
print(sum(time3)/len(time3))
num_tokens = {
        "num_tokens_baseline": sum(d["num_tokens_baseline"] for d in local_data)/len(time1),
        "num_tokens_dslr": sum(d["num_tokens_dslr"] for d in local_data)/len(time1),
        "num_tokens_pas_sen": sum(d["num_tokens_pas_sen"] for d in local_data)/len(time1)
    }
print(num_tokens)
print()

print(torch.version.cuda)

# model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# #Load the model and tokenizer from the local directory
# local_model_dir = Path.home().joinpath('mistral_model_fp16', '7B-Instruct-v0.3')
# local_model = AutoModelForCausalLM.from_pretrained(local_model_dir, torch_dtype=torch.float16)
# local_tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
# context = "Your context string goes here. Thistestcase can be any text you want to tokenize."

# # Tokenize the context
# tokens = local_tokenizer.encode(context)

# # Count the number of tokens
# token_count = len(tokens)

# print(f"Number of tokens: {token_count}")

# messages1 = [
#         {"role": "system", "content": "You are a helpful chatbot. Your task is to respond to the user’s question by looking into context."},
#         {"role": "user", "content": f"\n<s>Context: </s>\n\n<s>Based on this information, please answer the question: </s>"},
#     ]
# lcr_token = local_tokenizer.apply_chat_template(messages1, return_tensors='pt')
# lcr_token_length = lcr_token.shape[1]
# print(f"template token numbers: {lcr_token_length}")

# messages1 = [
#         {"role": "system", "content": "You are a helpful chatbot. Your task is to respond to the user’s question by looking into context."},
#         {"role": "user", "content": f"\n<s>Context: {context}</s>\n\n<s>Based on this information, please answer the question: </s>"},
#     ]
# lcr_token = local_tokenizer.apply_chat_template(messages1, return_tensors='pt')
# lcr_token_length = lcr_token.shape[1]
# print(f"template token numbers: {lcr_token_length}")

