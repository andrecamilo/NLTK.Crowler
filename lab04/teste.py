import nltk

from nltk.tokenize import sent_tokenize, word_tokenize

text = "eu gosto de muito de computador"

sents = sent_tokenize(text)
print(sents)

words =  word_tokenize(text)
print(words)

print(nltk.pos_tag(words))