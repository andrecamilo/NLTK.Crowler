# -*- coding: utf-8 -*-
import json
import coleta_tweets  # executa uma nova coleta de tweets

f_tweets = open('tweets.json', 'r')
f_novos = open('novos_tweets.txt', 'wb')

for linha in f_tweets:
    texto_tweet = json.loads(linha)['text'] + "\n"
    f_novos.write(texto_tweet.encode("utf8"))

f_novos.close()
f_tweets.close() 


