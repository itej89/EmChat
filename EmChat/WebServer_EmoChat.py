import sys
import base64
import re
import os
import flask
import threading, queue
import uuid 
from shutil import copyfile

try:
    import threading
except ImportError:
    import dummy_threading as threading


from EmoChat_Bot import *
import json

print("fmService : EMChat is loading model. . .")


Load_Model()

return_queue = queue.Queue()
request_queue = queue.Queue()
def from_main_thread_blocking():
    while True:
        message = request_queue.get() #blocks until an item is available
        sentenceID = message[0]
        EmosStats = predict_emotion(message[1])
        return_queue.put((sentenceID, EmosStats))

audioConvertor = threading.Thread(target=from_main_thread_blocking)
audioConvertor.start()
	
app = flask.Flask(__name__)
app.config["DEBUG"] = False

@app.route('/EMOCHAT', methods=['POST'])
def EMOCHAT():
	sentenceID = uuid.uuid1()
	params = flask.request.get_json()
	# sentence = params['sentence']
	data = (sentenceID, params)
	request_queue.put(data)
	message = return_queue.get()
	if message[0] == sentenceID :
		return json.dumps({ "TYPE" : "EMChat" , "DATA": message[1]})


@app.route('/WEB/EMOCHAT/<sentence>')
def WebEMOS(sentence):
	sentenceID = uuid.uuid1()
	data = (sentenceID, sentence)
	request_queue.put(data)
	message = return_queue.get()
	if message[0] == sentenceID :
		return "<h1>EMOCHAT Output</h1><p>"+json.dumps(message[1])+"</p>"



@app.route('/', methods=['GET'])
def home():
    return flask.render_template('basic.html', WebAPI="EMChat WebAPI : use /WEB/EMOCHAT/<sentence>",
	API = "EMChat API : use /EMOCHAT/<sentence>")


if __name__ == "__main__":
    print("fmService : EMChat is ready . . .!!")
    app.run(host='127.0.0.1', port=3581)

