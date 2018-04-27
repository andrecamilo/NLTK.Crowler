# -*- coding: utf-8 -*-
import tweepy
from tweepy import OAuthHandler

CONSUMER_API_KEY = ''
CONSUMER_API_SECRET = ''
ACCESS_TOKEN = ''
ACCESS_TOKEN_SECRET = ''

auth = tweepy.AppAuthHandler(CONSUMER_API_KEY, CONSUMER_API_SECRET)

api = tweepy.API(auth)
if (not api):
   print('Ocorreu um erro ao tentar conectar!')

