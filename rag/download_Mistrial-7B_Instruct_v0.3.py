from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
import sys


# mistral_models_path = Path.home().joinpath('mistral_model', '7B-Instruct-v0.3')
# mistral_models_path.mkdir(parents=True, exist_ok=True)

# Step 1: Download the model and tokenizer locally
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

# Define the path where the model will be downloaded
mistral_models_path = Path.home().joinpath('mistral_model', '7B-Instruct-v0.3')
mistral_models_path.mkdir(parents=True, exist_ok=True)

# Download the model files
snapshot_download(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    allow_patterns=["tokenizer.model"],#"params.json", "consolidated.safetensors", "tokenizer.model.v3"],
    local_dir=mistral_models_path
)

print(f"Model downloaded to {mistral_models_path}")



sys.exit() 


model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save the model and tokenizer locally
local_model_dir = Path.home().joinpath('mistral_model', '7B-Instruct-v0.3')
model.save_pretrained(local_model_dir)
tokenizer.save_pretrained(local_model_dir)

# Step 2: Load the model and tokenizer from the local directory
# local_model = AutoModelForCausalLM.from_pretrained(local_model_dir)
# local_tokenizer = AutoTokenizer.from_pretrained(local_model_dir)

# # Step 3: Create a pipeline using the local model and tokenizer
# chatbot = pipeline("text-generation", model=local_model, tokenizer=local_tokenizer)

# # Define the messages
# messages = [
#     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
#     {"role": "user", "content": "Who are you?"},
# ]

# # Generate a response
# response = chatbot(messages)
# print(response)
