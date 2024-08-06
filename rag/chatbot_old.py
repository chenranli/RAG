from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from transformers import AutoModelForCausalLM, AutoTokenizer 

mistral_models_path = "MISTRAL_MODELS_PATH"
 
tokenizer = MistralTokenizer.v1()
 
completion_request = ChatCompletionRequest(messages=[UserMessage(content="Explain Machine Learning to me in a nutshell.")])
 
tokens = tokenizer.encode_chat_completion(completion_request).tokens

print(tokens)
print(type(tokens))

device = "cuda" # the device to load the model onto

#model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", token="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")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", token="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")
messages = [
    # {"role": "user", "content": "What is your favourite condiment?"},
    # {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    # {"role": "user", "content": "Do you have mayonnaise recipes?"}
    {"role": "user", "content": "Explain Machine Learning to me in a nutshell."}
]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
print(encodeds)
print(type(encodeds))


local_model_dir_tokenizer = "../../mistral_models/7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(local_model_dir_tokenizer)

