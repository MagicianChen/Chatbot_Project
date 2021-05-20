# Chatbot_Project
## Overview
I develop this procject to learn more about chatbot system. It consists of two parts: single-turn and multi-turn. For single-turn demo, i try three strategies: word-level matching, embedding-level matching and tree-based matching. As for multi-turn demo, i apply graph and deep learning to achieve it. The content will be continuouly updated.
## key topic
1. single-turn demo
2. multi-turn demo

## Single-turn
### Requirement
``` 
python == 3.x  
Flase  
nltk  
fasttext  
jieba  
```
### Test Demo
run the ques_ans.py to test the demo, you can change the matching mode, data path and word2vec model file path in this file.
```
python ques_ans.py
```
if you want to connect it with your own chatbot app, you can build the flask server through QA_system.py
```
python QA_system.py  
```
