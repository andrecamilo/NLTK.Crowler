from nltk.tokenize import sent_tokenize
frases = sent_tokenize("gastei trinta reais no almoço do dia 30 de março")
print(frases[:10])
from nltk.tokenize import word_tokenize
palavras = word_tokenize(frases[0])
print(palavras)
