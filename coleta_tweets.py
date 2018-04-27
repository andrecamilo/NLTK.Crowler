# -*- coding: utf-8 -*-
from autoriza_app import api
import jsonpickle
import tweepy

consulta = 'place:n'	

numMaxTweets = int(input('Informe o número máximo de tweets desejados: '))

contadorTweets = 0

with open('tweets.json', 'w') as f:

    for tweet in tweepy.Cursor(api.search, q = consulta).items(numMaxTweets) :         
        f.write(jsonpickle.encode(tweet._json, unpicklable=False) + '\n')
        contadorTweets += 1

    print("Foram coletados " + str(contadorTweets) + " tweets.")

