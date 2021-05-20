import os
import jieba
import re

jieba.initialize()
punc = "[^0-9A-Za-z\u4e00-\u9fa5] "

# pre-processing the corpus, split the candidates into word-level for bleu matching mode
def preprocessing(file_name):
    path = './data'
    file_path = os.path.join(path, file_name)
    new_file_path = os.path.join(path, 'new_'+file_name)
    with open(file_path,'r',encoding='utf-8') as f1:
        with open(new_file_path,'w',encoding='utf-8') as f2:
            for line in f1:
                data = eval(line.strip())
                data['candidate'] = [list(jieba.cut(re.sub(punc, "", text))) for text in data['candidate']]
                f2.write(str(data)+'\n')


if __name__ == '__main__':
    preprocessing('Corpus_example.txt')
