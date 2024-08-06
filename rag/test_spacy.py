# !pip install https://huggingface.co/spacy/en_core_web_sm/resolve/main/en_core_web_sm-any-py3-none-any.whl

# Using spacy.load().
# import spacy
# nlp = spacy.load("en_core_web_sm")

# Importing as module.
#import en_core_web_sm
import spacy
import time

start_time  = time.time()

#nlp = en_core_web_sm.load()

# passage = """
# This is a sample passage. It contains multiple sentences. 
# Some sentences might be short. Others could be longer and more complex, 
# containing multiple clauses or phrases.
# """
passage = """\
J.K. Rowling is the author of the Harry Potter series.You can contact me at john.doe@example.com.My salary is 2000.23 USD. Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary\
"""
# passage = """\
# Game players were not the only ones to notice the violence in this game; US Senators Herb Kohl and Joe Lieberman convened a Congressional hearing on December 9, 1993 to investigate the marketing of violent video games to children.[e] While Nintendo took the high ground with moderate success, the hearings led to the creation of the Interactive Digital Software Association and the Entertainment Software Rating Board, and the inclusion of ratings on all video games. With these ratings in place, Nintendo decided its censorship policies were no longer needed.\n
# """

# passage = f"""On release, Twilight Princess was considered to be the greatest Zelda game ever made by many critics including writers for 1UP.com, Computer and Video Games, Electronic Gaming Monthly, Game Informer, GamesRadar, IGN and The Washington Post. Game Informer called it "so creative that it rivals the best that Hollywood has to offer". GamesRadar praised Twilight Princess as "a game that deserves nothing but the absolute highest recommendation". Cubed3 hailed Twilight Princess as "the single greatest videogame experience". Twilight Princess's graphics were praised for the art style and animation, although the game was designed for the GameCube, which is technically lacking compared to the next generation consoles. Both IGN and GameSpy pointed out the existence of blurry textures and low-resolution characters. Despite these complaints, Computer and Video Games felt the game's atmosphere was superior to that of any previous Zelda game, and regarded Twilight Princess's Hyrule as the best version ever created. PALGN praised the game's cinematics, noting that "the cutscenes are the best ever in Zelda games". Regarding the Wii version, GameSpot's Jeff Gerstmann said the Wii controls felt "tacked-on", although 1UP.com said the remote-swinging sword attacks were "the most impressive in the entire series". Gaming Nexus considered Twilight Princess's soundtrack to be the best of this generation, though IGN criticized its MIDI-formatted songs for lacking "the punch and crispness" of their orchestrated counterparts. Hyper's Javier Glickman commended the game for its "very long quests, superb Wii controls and being able to save anytime". However, he criticised it for "no voice acting, no orchestral score and slightly outdated graphics"."""

# Process the passage
# doc = nlp(passage)
# doc = nlp(passage.strip())

# # Extract sentences
# sentences = list(doc.sents)

# end_time = time.time()
# print(f"model time: {end_time - start_time} seconds\n")

# # Print each sentence
# for sentence in sentences:
#     print(f'"{sentence.text.strip()}"')


#------------------------------
start_time  = time.time()
#test sentencizer
nlp2 = spacy.blank("en")
# Add the sentencizer to the pipeline
nlp2.add_pipe("sentencizer")
# Create a spaCy document
#doc = nlp(passage.strip())    
# Extract sentences
doc = nlp2(passage.strip())
sentences = [sent for sent in doc.sents]
end_time = time.time()
print(f"sentencizer time: {end_time - start_time} seconds\n")

for sentence in sentences:
    print(f'"""{sentence.text.strip()}""",')
    #print(sentence)
#print(f"typeof(sentence): {type(sentences[0])}") Its type is: <class 'spacy.tokens.span.Span'>
#print