#Create a bot that reads messages from a twitch chat

import socket
import json
from emoji import demojize
import logging
import pandas as pd
from datetime import datetime
import re



server = "irc.chat.twitch.tv"
port = 6667
nickname = 'testbot'
token = "oauth:lzpec51zyiyrl9k1ayx1hl3r3m09li"
channel = '#lec'

logging.basicConfig(level=logging.DEBUG,format = '%(asctime)s | %(message)s', datefmt= '%Y-%m-%d_%H:%M:%S', handlers= [logging.FileHandler('chat.log', encoding= 'utf-8')])

data = {}
data["message"] = []

sock = socket.socket()

sock.connect((server,port))

sock.send(f"PASS {token}\n".encode('utf-8'))
sock.send(f"NICK {nickname}\n".encode('utf-8'))
sock.send(f"JOIN {channel}\n".encode('utf-8'))

response = sock.recv(2048).decode('utf-8')

print(response)

while True:
    response = sock.recv(2048).decode('utf-8')

    if response.startswith('PING'):
        sock.send('PONG\n'.encode('utf-8'))

    elif len(response) > 0:\
        #Logging
        #data["message"].append({"username":response.})
        logging.info(demojize(response))
        print(demojize(response))
        #print(response)

sock.close()

