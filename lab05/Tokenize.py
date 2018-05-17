import nltk
texto = "Acabei de almoçar e gastei 19 reais"
frases = nltk.tokenize.sent_tokenize(texto)
tokens = nltk.word_tokenize(texto)
classes = nltk.pos_tag(tokens)
print(classes)