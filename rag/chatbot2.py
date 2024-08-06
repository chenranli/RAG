from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import torch
import time

local_model_dir = Path.home().joinpath('mistral_model_fp16', '7B-Instruct-v0.3')
#Load the model and tokenizer from the local directory
local_model = AutoModelForCausalLM.from_pretrained(local_model_dir, torch_dtype=torch.float16)
local_tokenizer = AutoTokenizer.from_pretrained(local_model_dir)

# tokenizer = MistralTokenizer.v1()
# completion_request = ChatCompletionRequest(messages=[UserMessage(content="Explain Machine Learning to me in a nutshell.")])
# tokens = tokenizer.encode_chat_completion(completion_request).tokens
# print(tokens)
# print(type(tokens))

# messages = [
#     # {"role": "user", "content": "What is your favourite condiment?"},
#     # {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
#     # {"role": "user", "content": "Do you have mayonnaise recipes?"}
#     {"role": "user", "content": "Explain Machine Learning to me in a nutshell."}
# ]

# encodeds = local_tokenizer.apply_chat_template(messages, return_tensors="pt")
# print(encodeds)
# print(type(encodeds))
# print()


passages = ["""The first degrees from the college were awarded in 1849. The university was expanded with new buildings to accommodate more students and faculty. With each new president, new academic programs were offered and new buildings built to accommodate them. The original Main Building built by Sorin just after he arrived was replaced by a larger "Main Building" in 1865, which housed the university's administration, classrooms, and dormitories. Beginning in 1873, a library collection was started by Father Lemonnier. By 1879 it had grown to ten thousand volumes that were housed in the Main Building.""",
    #"""This Main Building, and the library collection, was entirely destroyed by a fire in April 1879, and the school closed immediately and students were sent home. The university founder, Fr. Sorin and the president at the time, the Rev. William Corby, immediately planned for the rebuilding of the structure that had housed virtually the entire University. Construction was started on the 17th of May and by the incredible zeal of administrator and workers the building was completed before the fall semester of 1879. The library collection was also rebuilt and stayed housed in the new Main Building for years afterwards. Around the time of the fire, a music hall was opened. Eventually becoming known as Washington Hall, it hosted plays and musical acts put on by the school. By 1880, a science program was established at the university, and a Science Hall (today LaFortune Student Center) was built in 1883. The hall housed multiple classrooms and science labs needed for early research at the university.""",
    #"""In 1899, the university opened a national design contest for the new campus. The renowned Philadelphia firm Cope & Stewardson won unanimously with its plan for a row of Collegiate Gothic quadrangles inspired by Oxford and Cambridge Universities. The cornerstone of the first building, Busch Hall, was laid on October 20, 1900. The construction of Brookings Hall, Ridgley, and Cupples began shortly thereafter. The school delayed occupying these buildings until 1905 to accommodate the 1904 World's Fair and Olympics. The delay allowed the university to construct ten buildings instead of the seven originally planned. This original cluster of buildings set a precedent for the development of the Danforth Campus; Cope & Stewardson’s original plan and its choice of building materials have, with few exceptions, guided the construction and expansion of the Danforth Campus to the present day.""",
    ]
context = "".join(passages)    
question = "Which structure was the first used for the purposes of the college?"

messages = [
    {"role": "system", "content": "You are a helpful chatbot. Your task is to respond to the user’s question by looking into context."},
    {"role": "user", "content": f"\n<s>Context: {context}</s>\n\n<s>Based on this information, please answer the question: {question}</s>"},
]

device = 0 if torch.cuda.is_available() else -1
chatbot = pipeline("text-generation", model=local_model, tokenizer=local_tokenizer, device=device)

def generate_response(prompt, max_length=1000):
    # Generate a response
    # response = chatbot(prompt, max_length=max_length, num_return_sequences=1, temperature=0.7, top_p=0.95, do_sample=True)
    input_string = ""
    for item in prompt:
        if item["role"] == "system":
            input_string += f"<s>{item['content']}</s>\n"
        elif item["role"] == "user":
            input_string += f"{item['content']}\n"

    print(f"Real prompt: {input_string}")
    # Tokenize the input prompt to get the number of input tokens
    input_tokens = local_tokenizer.apply_chat_template(prompt, return_tensors='pt')
    input_token_length = input_tokens.shape[1]

    # Generate a response using the pipeline
    # The pipeline handles tokenization internally, so we don't need to tokenize separately
    response = chatbot(
        prompt, 
        max_length=max_length, 
        num_return_sequences=1, 
        #temperature=0.7, 
        #top_p=0.95, 
        #do_sample=True,
        truncation=True  # Enable truncation within the pipeline
    )
    # Extract the generated text
    generated_text = response[0]['generated_text']
    # Tokenize the generated response to get the number of output tokens
    output_tokens = local_tokenizer.apply_chat_template(generated_text, return_tensors='pt')
    output_token_length = output_tokens.shape[1]
    # Print the lengths of input and output tokens
    print(f"Input token length: {input_token_length}")
    print(f"Output token length: {output_token_length}\n")
    
    return generated_text

start_time = time.time()
response = generate_response(messages)
end_time = time.time()
print(f"response time: {end_time - start_time} seconds")
print(f"context word count: {len(context.split())}")

# Test the model
#prompt = "Explain the concept of machine learning in simple terms."

#print(f"Prompt: {messages}\n")
#print(f"Response: {response}\n")
print(f"Answer: {response['response']}")

#response = chatbot(messages)
#print(response)