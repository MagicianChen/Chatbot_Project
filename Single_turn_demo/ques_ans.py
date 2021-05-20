# -*- coding:utf-8 -*-
# -*- created by: mo -*-
import json
import time
import datetime
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import re
import random
import jieba
import fasttext

jieba.initialize()
punc = "[^0-9A-Za-z\u4e00-\u9fa5] "

class Chatbot():
    def __init__(self,mode, corpus_path, w2v_path):
        self.mode = mode
        # load the training corpus
        with open(corpus_path,'r',encoding='utf-8') as f:
            self.data = []
            for i in f:
                self.data.append(eval(i.strip()))

        self.data_size = len(self.data)

        if mode == 'embedding':
            self.model = fasttext.load_model(w2v_path)  # load fasttext model
            self.corpus_vec = []
            for item in self.data:
                self.corpus_vec.append(self.WordToVec(item['ques']))  # obtain word embeeding at first


    def cosine_similarity(self, a, b):
        # calculate the semantic similarity between question and corpus
        a = np.array(a).reshape((len(a),len(b[0])))
        a_norm = np.linalg.norm(x=a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(x=b, axis=1)
        results = np.dot(a, b.T) / (a_norm * b_norm)

        return results


    def record(self,ans, ques, src):
        # record the QA history for improving your system
        f = open('./record.txt', 'a', encoding='utf-8')
        f.write(ques + '\t' + ans + '\t' + src + '\n')
        f.close()


    def WordToVec(self,text):
        # if use word embedding matching, you can load word embedding model here.

        lower_sentence = text[0].lower()
        lower_sentence = re.sub(punc, "", lower_sentence)  # remove punctuation
        word_tokens = list(jieba.cut(lower_sentence))

        word_vec = []
        for word in word_tokens:
            word_vec.append(self.model[word.strip()])

        # average a list of word vectors, obtain the sentence vector
        vector = np.mean(word_vec, axis=0).reshape((1,-1))

        return vector


    def corpus_bleu(self, ques):
        scores = []
        split_ques = list(jieba.cut(ques))

        # default weight of 4-bleu [0.25, 0.25, 0.25, 0.25], avoid a too short length question.
        weight = [0,0,0,0]
        if len(split_ques) < 4:
            for i in range(len(split_ques)):
                weight[i] = 1 / len(split_ques)
        else:
            weight = [0.25, 0.25, 0.25, 0.25]
        weight = tuple(weight)

        for item in self.data:
            scores.append(sentence_bleu(item['candidate'], split_ques,weights=weight))

        return scores


    def ques_ans(self,text):
        if self.mode == 'embedding': # use word embedding matching
            ques_vec = self.WordToVec(text)
            matching_results = self.cosine_similarity(self.corpus_vec, ques_vec)

            # Set the threshold value to judge the fit of the match
            if matching_results.max() >= 0.5:
                #print(matching_results)
                return self.data[np.argmax(matching_results)]['answer']  # return matcching result
            else:
                return 'I can not answer this question'  # or use dialogue generation model
        elif self.mode == 'word': # based on BLEU matching
            # calculate candidates for each data
            match_score = self.corpus_bleu(text)
            max_idx = np.argmax(match_score)

            # Set the threshold value to judge the fit of the match
            if match_score[max_idx] > 0.3:
                return self.data[max_idx]['answer']  # return matcching result
            else:
                #  or use dialogue generation model, such as Tuling API
                return 'I can not answer this question'



if __name__ == '__main__':
    corpus_path = './data/new_Corpus_example.txt' # change the training corpus path
    w2v_path = './word2vec/cc.zh.300.bin'  # change the word2vec model path
    chatbot = Chatbot('embedding', corpus_path, w2v_path)

    while True:
        ques = input('user: ')
        ans = chatbot.ques_ans(ques)
        print('Chatbot: '+ans + '\n')
