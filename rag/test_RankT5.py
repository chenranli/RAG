from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from utils import rerank_passages
import time

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Soyoung97/RankT5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("Soyoung97/RankT5-base")

# Prepare data

# passages = ["Architecturally, the school has a Catholic character.",
# "Atop the Main Building's gold dome is a golden statue of the Virgin Mary.",
# """Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes".""",
# "Next to the Main Building is the Basilica of the Sacred Heart.",
# "Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection.",
# "It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858.",
# "At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary"]

titles = ['University_of_Notre_Dame', 'University_of_Notre_Dame', 'Washington_University_in_St._Louis']

passages = ["""The first degrees from the college were awarded in 1849. The university was expanded with new buildings to accommodate more students and faculty. With each new president, new academic programs were offered and new buildings built to accommodate them. The original Main Building built by Sorin just after he arrived was replaced by a larger "Main Building" in 1865, which housed the university's administration, classrooms, and dormitories. Beginning in 1873, a library collection was started by Father Lemonnier. By 1879 it had grown to ten thousand volumes that were housed in the Main Building.""",
    """This Main Building, and the library collection, was entirely destroyed by a fire in April 1879, and the school closed immediately and students were sent home. The university founder, Fr. Sorin and the president at the time, the Rev. William Corby, immediately planned for the rebuilding of the structure that had housed virtually the entire University. Construction was started on the 17th of May and by the incredible zeal of administrator and workers the building was completed before the fall semester of 1879. The library collection was also rebuilt and stayed housed in the new Main Building for years afterwards. Around the time of the fire, a music hall was opened. Eventually becoming known as Washington Hall, it hosted plays and musical acts put on by the school. By 1880, a science program was established at the university, and a Science Hall (today LaFortune Student Center) was built in 1883. The hall housed multiple classrooms and science labs needed for early research at the university.""",
    """In 1899, the university opened a national design contest for the new campus. The renowned Philadelphia firm Cope & Stewardson won unanimously with its plan for a row of Collegiate Gothic quadrangles inspired by Oxford and Cambridge Universities. The cornerstone of the first building, Busch Hall, was laid on October 20, 1900. The construction of Brookings Hall, Ridgley, and Cupples began shortly thereafter. The school delayed occupying these buildings until 1905 to accommodate the 1904 World's Fair and Olympics. The delay allowed the university to construct ten buildings instead of the seven originally planned. This original cluster of buildings set a precedent for the development of the Danforth Campus; Cope & Stewardson’s original plan and its choice of building materials have, with few exceptions, guided the construction and expansion of the Danforth Campus to the present day.""",
    ]

# Create input pairs
# def prepare_input(sentence, passage):
#     return f"Query: {sentence} Document: {passage}"

# input_pairs = [prepare_input(sentence, passage) for passage in passages]

# # Tokenize inputs
# inputs = tokenizer(input_pairs, return_tensors="pt", padding=True, truncation=True)

# # Generate ranking scores
# with torch.no_grad():
#     outputs = model.generate(**inputs, return_dict_in_generate=True, output_scores=True, max_new_tokens=1)

# scores = outputs.scores[0][:, 1]  # Assuming 1 is the token ID for "true" or high relevance

# # Rank passages
# ranked_passages = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
question = "Which structure was the first used for the purposes of the college?"

start_time  = time.time()
ranked_passages = rerank_passages(question, titles, passages, model, tokenizer)
end_time = time.time()
print(f"reranking time: {end_time - start_time} seconds")

# Print ranked passages
for i, (score, passage) in enumerate(ranked_passages, 1):
    print(f"{i}. Score: {score:.4f} - {len(passage.split())} - {passage}")
