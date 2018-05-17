#-*- coding: utf-8 -*-
from chatterbot.trainers import ListTrainer
from chatterbot import ChatBot
import os

bot = ChatBot('teste')#, read_only=True
bot.set_trainer(ListTrainer)

for arq in os.listdir('arq'):
    chats = open('arq/' + arq, 'r').readlines()
    bot.train(chats)

while True:
    resq = input('vc: ')

    resp = bot.get_response(resq)
    print('bot: ' + str(resp))