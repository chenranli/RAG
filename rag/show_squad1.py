import argparse
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

sentences = ["This is an example sentence", "Each sentence is converted"]

# model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# embeddings = model.encode(sentences)
# print(embeddings)
# print("embedding length: ", len(embeddings[0]))
# print(f"embedding type: {type(embeddings[0])}")

# Load SQuAD2.0
squad = load_dataset('squad')

# Print the first example from the validation set of SQuAD2.0
# print(squad_v2['train'][0])
# print(f"type of element squad_v2['train'][0]: {type(squad_v2['train'][0])}")
# print(f"type of context:{type(squad_v2['train'][0]['context'])}")

# Print the first example from the training set
#print(squad_v2['train'][0])
# print(40099)
# print(squad_v2['train'][40099])
# print(f"-------------------------------------------------------")
# print(40100)
# print(squad_v2['train'][40100])
# print(f"-------------------------------------------------------")
# print(40101)
# print(squad_v2['train'][40101])
# print(f"-------------------------------------------------------")
# print(40102)
# print(squad_v2['train'][40102])
# print(f"-------------------------------------------------------")


print(f"length of squad_v2 training dataset: {len(squad['train'])}\n")
print(f"length of squad_v2 training dataset: {len(squad['validation'])}\n")
# Create the parser without a description
parser = argparse.ArgumentParser()
# Add arguments
parser.add_argument("arg1", type=int, help="data index (integer)")
# Parse the arguments
args = parser.parse_args()
print(f"{squad['train'][args.arg1]}\n")
print(f"context: {squad['train'][args.arg1]['context']}\n")
print(f"question: {squad['train'][args.arg1]['question']}\n")
print(f"answers: {squad['train'][args.arg1]['answers']}\n")
print(f"answers_text: {squad['train'][args.arg1]['answers']['text']}\n")

# print(f"{squad['validation'][args.arg1]}\n")
# print(f"context: {squad['train'][args.arg1]['context']}\n")
# print(f"question: {squad['train'][args.arg1]['question']}\n")
# print(f"answers: {squad['train'][args.arg1]['answers']}\n")
# print(f"answers_text: {squad['train'][args.arg1]['answers']['text']}\n")