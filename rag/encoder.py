import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

# # Load SQuAD2.0 dataset
# squad_dataset_train = load_dataset("squad", split="train")

# Load the SQuAD 1.1 dataset
squad_dataset = load_dataset('squad')

# Access the train and validation splits
train_dataset = squad_dataset['train']
#validation_dataset = squad_dataset['validation']

# Setup Sentence Transformer for embeddings
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cuda')

#count = 0
print(f"how many subsets do squad1.1 have: {len(squad_dataset)}")


#Extract all passages (contexts)
all_contexts = train_dataset['context']
titles = train_dataset['title']
print(f" Total contexts in train set: {len(all_contexts)}")
unique_tuples = set(zip(all_contexts, titles))
print(f"Unique (context, title) tuples in train set: {len(unique_tuples)}") # = 18891

# count = 100
# for (context,title) in unique_tuples:
#     #print(context)
#     print(title)
#     count = count - 1
#     if count == 0:
#         break

# unique_contexts = list(set(all_contexts)) 
# print(f"Unique contexts in train set: {len(unique_contexts)}") # = 18891

# # Extract all passages (contexts)
# print(f"validation set length: {len(validation_dataset)}") = 10570
# print(f"validation set type: {type(validation_dataset)}") = <class 'datasets.arrow_dataset.Dataset'>
# all_contexts1 = train_dataset = validation_dataset["context"]
# print(f" Total contexts in validation set: {len(all_contexts1)}")
# # Remove duplicates to get unique passages
# unique_contexts1 = list(set(all_contexts1))
# print(f"Unique contexts in validation set: {len(unique_contexts1)}")


# Prepare data for saving
data_to_save = []
for (context,title) in unique_tuples:
    context_embedding = model.encode(context).tolist()  # Convert to list for JSON serialization
    data_to_save.append({
        'context': context,
        'embedding': context_embedding,
        'title': title
    })
    # count += 1
    # if count == 100:
    #     print('program terminated.')
    #     break

# Save data to a local file
output_file = 'squad1.1_embeddings.json'
with open(output_file, 'w') as f:
    json.dump(data_to_save, f)

print(f"Embeddings and contexts saved to {output_file}")
