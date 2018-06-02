from nltk.corpus import stopwords
stopwords_lista = stopwords.words('portuguese')
print(stopwords_lista)
nao_stopwords = [w for w in words if not w.lower() in stopwords_lista]
print(nao_stopwords)