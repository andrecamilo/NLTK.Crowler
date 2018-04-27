# -*- coding: utf-8 -*-
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier
import pickle

def formatar_sentenca(sentenca):
   return {palavra: True for palavra in word_tokenize(sentenca)}

f_pos = open('corpus_positivo.txt', 'rb')
positivos = f_pos.read().splitlines()
f_pos.close()
f_neg = open('corpus_negativo.txt', 'rb')
negativos = f_neg.read().splitlines()
f_neg.close()

dados_treinamento = []

for positivo in positivos:
   dados_treinamento.append([formatar_sentenca(positivo.decode("utf8").lower()), "positivo"])
for negativo in negativos:
   dados_treinamento.append([formatar_sentenca(negativo.decode("utf8").lower()), "negativo"])
   
modelo = NaiveBayesClassifier.train(dados_treinamento)	

with open('modelo.obj', 'wb') as f:
    modelo_serial = pickle.dump(modelo, f)
    print('Modelo classificador treinado e armazenado em modelo.obj')

