#Create a bot that reads messages from a twitch chat

import socket
import json
from emoji import demojize

server = "irc.chat.twitch.tv"
port = 6667
nickname = 'testbot'
token = "oauth:lzpec51zyiyrl9k1ayx1hl3r3m09li"
channel = '#yassuo'

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
        print(demojize(response))
        #print(response)

sock.close()