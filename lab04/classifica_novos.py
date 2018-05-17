# -*- coding: utf-8 -*-
from nltk.tokenize import word_tokenize
from datetime import datetime
import pickle

def formatar_sentenca(sentenca):
   return {palavra: True for palavra in word_tokenize(sentenca)}

with open('modelo.obj', 'rb') as f:
    modelo = pickle.load(f) 

f_resultado = open('resultado.txt', "ab")
data_hora = '\n' + str(datetime.now()) + '\n'
f_resultado.write(data_hora.encode("utf8"))

f_sentenca = open('novos_tweets.txt', encoding="utf8")
sentencas = f_sentenca.read().splitlines()

for sentenca in sentencas:
   sentimento = modelo.classify(formatar_sentenca(sentenca.lower()))
   texto_resultado = '(' + sentimento + ') ' + sentenca + '\n'
   f_resultado.write((texto_resultado.encode("utf8")))

f_resultado.close()
f_sentenca.close()   

print('A análise de sentimento dos novos tweets foi armazenada em resultado.txt')

