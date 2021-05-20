# Chatbot_Project
## Overview
I develop this procject to learn more about chatbot system. It consists of two parts: single-turn and multi-turn. For single-turn demo, i try three strategies: word-level matching, embedding-level matching and tree-based matching. As for multi-turn demo, i apply graph and deep learning to achieve it. The content will be continuouly updated.
## key topic
1. single-turn demo
2. multi-turn demo

## Single-turn
### Description
The matching mode chatbot is more suitable for task-oriented QA, since it need a high controllability. 
#### Word-level
I achieve it by BLEU score, which is totally based on word character and compute quickly. However, it can't consider the semantic level matching. Thus you need to prepare many candidates to confirm the matching accuracy of different type of questions.
#### Embedding-level
This approach will consider more semantic level information, which utilize word-embedding model such as fasttext or Glove. In my program, i use fasttext model to calculate cosine similarity to measure the matching degree. In order to speed up the calculation, i calculate the word vector and store in .txt file for each question before running.
### Requirement
``` 
python == 3.x  
Flase  
nltk  
fasttext  
jieba  
```
### Prepare data
prepare question answer pair as a txt file and store in './data/'. The key 'candidate' is used to enhance the matching performance of BLEU score.
```
{  
   'ques': question,
   'answer': answer,
   'candidate': [candidate matching questions]
}
```
Example:
```
{
    'ques': 'who are you',
    'answer': 'I am chatbot',
    'candidate': ['Who', 'what is your name', 'who you are']
}
```
### Test Demo
split the word for training data at first
```
python preprocess.py
```
run the ques_ans.py to test the demo, you can change the matching mode, data path and word2vec model file path in this file. Before running, you need to download the fasttext pre-trained model: [fasttext](https://fasttext.cc/)
```
python ques_ans.py
```
if you want to connect it with your own chatbot app, you can build the flask server through QA_system.py
```
python QA_system.py  
```

## Multi-turn
Working on it...
