# -*- coding: utf-8 -*-

import json
import requests
from flask import Flask, request, make_response,render_template, Response
import time
import xmltodict
import os
from ques_ans import *
import datetime

app = Flask(__name__)
corpus_path = './data/new_Corpus_example.txt' # change the training corpus path
w2v_path = './word2vec/cc.zh.300.bin'  # change the word2vec model path
chatbot = Chatbot('word', corpus_path, w2v_path) # 'word': word-level matching based on bleu; '
                                                # embedding': embedding-level matching based on word embedding model


@app.route('/applet', methods=['POST'])
def applet():
    if request.method == 'POST':
        # Receive data from app
        data = request.data
        data = eval(data.decode('utf-8'))

        if data['action'] == 'login':
            return str(time.asctime(time.localtime(time.time())))
        elif data['action'] == 'text':
            ques = data['ques']
            ans = chatbot.ques_ans(ques)
            return {'type': 'text','text': ans}
        elif data['action'] == 'voice':
            # voice recognize api
            pass



if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8000,threaded=True)