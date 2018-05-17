import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import cross_val_predict

dataset = pd.read_csv('sentimento.csv')
dataset.count()

frases = dataset['Text'].values
classes = dataset['Classificacao'].values

freq_frases = vectorizer.fit_transform(frases)
modelo = MultinomialNB()
modelo.fit(freq_frases,classes)

testes = ['Esse governo está no início, vamos ver o que vai dar',
         'Estou muito feliz com o governo de Minas esse ano',
         'O estado de Minas Gerais decretou calamidade financeira!!!',
         'A segurança desse país está deixando a desejar',
         'O governador de Minas é do PT']

freq_testes = vectorizer.transform(testes)
modelo.predict(freq_testes)

# array(['Neutro', 'Neutro', 'Negativo', 'Negativo', 'Neutro']
# resultados = cross_val_predict(modelo, freq_tweets, sentimento, cv=10)
# metrics.accuracy_score(classes,resultados)
 