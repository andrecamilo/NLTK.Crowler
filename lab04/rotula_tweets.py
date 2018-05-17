# -*- coding: utf-8 -*-
import json

f_tweets = open('tweets.json', 'r')

f_pos = open('corpus_positivo.txt', 'wb')
f_neg = open('corpus_negativo.txt', 'wb')

for linha in f_tweets:

    tweet = json.loads(linha)

    texto_tweet = tweet['text'] + "\n"	
    print('\n-------------\nTWEET: ' + texto_tweet + '(p/n): ' )
    sentimento = input()
	
    if (sentimento == 'p'):
        f_pos.write(texto_tweet.encode("utf8"))
    elif (sentimento == 'n'):
        f_neg.write(texto_tweet.encode("utf8"))

f_pos.close()
f_neg.close()
f_tweets.close() 


