# -*- coding: utf-8 -*-
import tweepy
from tweepy import OAuthHandler

CONSUMER_API_KEY = "Gpj7COqxcp1No74bcgWldDz7w"
CONSUMER_API_SECRET = "sapFR6jypP8QRls6NTeAc2zVAMOOYmOnpiUAmumLdKmg42zNVc"
ACCESS_TOKEN = "50280987-LQzR3DJo8kchyMa535ZAZUd14L2KfH1YF9QsoGoXj"
ACCESS_TOKEN_SECRET = "PpZuShzI20hgUbKv7rSeUCcSTdlVwfZ34VXSxu3tgViYV"


auth = tweepy.AppAuthHandler(CONSUMER_API_KEY, CONSUMER_API_SECRET)

api = tweepy.API(auth)
if (not api):
   print('Ocorreu um erro ao tentar conectar!')

