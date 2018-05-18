#-*- coding: utf-8 -*-
from chatterbot.trainers import ListTrainer
from chatterbot import ChatBot
import speech_recognition as sr
import pyttsx3
import os

bot = ChatBot('Logo')
bot.set_trainer(ListTrainer)

speak = pyttsx3.init('sapi5')

def Speak(text):
    speak.say(text)
    speak.runAndWait()

for arq in os.listdir('arq'):
    chats = open('arq/' + arq, 'r').readlines()
    bot.train(chats)

r = sr.Recognizer()

with sr.Microphone() as s:
    r.adjust_for_ambient_noise(s)

while True:
    print("diga alguma coisa")
    audio = r.listen(s)
    speech = r.recognize_google(audio)
    response = bot.get_response(speech)
    print('vc: ' + speech)
    print('bot: ' + response)

    Speak(response)




while True:
    resq = input('vc: ')

    resp = bot.get_response(resq)
    print('bot: ' + str(resp))