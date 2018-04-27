# -*- coding: utf-8 -*-
from nltk.tokenize import word_tokenize
import pickle

def formatar_sentenca(sentenca):
   return {palavra: True for palavra in word_tokenize(sentenca)}

with open('modelo.obj', 'rb') as f:
    modelo = pickle.load(f) 

sentenca = input('\nDigite a sentença a ser analisada: ')

while sentenca != '':
   sentimento = modelo.classify(formatar_sentenca(sentenca.lower()))
   print('O sentimento desta sentença é ' + sentimento + '.\n')
   sentenca = input('Digite a sentença a ser analisada: ')




