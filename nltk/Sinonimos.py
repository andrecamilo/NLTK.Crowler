import nltk
from nltk.corpus import wordnet
sinonimo = wordnet.synsets("importante")
print(sinonimo)
print(sinonimo[0].definition()) 
print(sinonimo[0].examples())
