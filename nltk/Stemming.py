from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
stemmer = SnowballStemmer('portuguese')
sample_sentence = "Gastei trinta reais no almoço"
sample_words = word_tokenize(sample_sentence)
stems = [stemmer.stem(w) for w in sample_words]
print(stems)