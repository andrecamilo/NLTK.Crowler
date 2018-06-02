import nltk
nltk.download('wordnet')       
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
lemmatizer = WordNetLemmatizer()
pos_tags = nltk.pos_tag("paguei a conta da loja")
lemmas = []
for w in pos_tags:
    if w[1].startswith('J'):
        pos_tag = wordnet.ADJ
    elif w[1].startswith('V'):
        pos_tag = wordnet.VERB
    elif w[1].startswith('N'):
        pos_tag = wordnet.NOUN
    elif w[1].startswith('R'):
        pos_tag = wordnet.ADV
    else:
        pos_tag = wordnet.NOUN
    lemmas.append(lemmatizer.lemmatize(w[0], pos_tag))
print(lemmas)