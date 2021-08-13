# coding=utf-8
# import os
import json
from nltk.corpus import stopwords
import nltk.stem
import string
from gensim.models import Word2Vec
import numpy as np
import gensim


def word2vec_model(text_processed_filename, model_filename, intermediate_vb_filename):
    # 加载预处理后的数据
    text_processed = np.load('../dataset/Amazon/result/lexicon_semi/' + text_processed_filename,
                             allow_pickle=True).tolist()
    # 1. 训练模型 并进行存储，之前非常疑惑输入进word2vec的sentences是什么，现在由
    # [博客](https://towardsdatascience.com/a-beginners-guide-to-word-embedding-with-gensim-word2vec-model-5970fa56cc92)
    # 可以知道，输入进入word2vec的sentences就是嵌套的`list`
    # 2. 存储word2vec，以及使用sentence generator（当数据过大的时候，需要使用`yield`来慢慢生成sentence）
    # 参考 [博客](https://blog.csdn.net/u010665216/article/details/78709018)
    # train the model
    model = Word2Vec(text_processed, min_count=1, vector_size=50, workers=3, window=3, sg=1)
    # 存储模型
    model.save('../dataset/Amazon/model/' + model_filename)

    # generate the list of seed words
    wordsTxtFileName0 = 'positive-words.txt'
    wordsTxtFileName1 = 'negative-words.txt'
    wordsTxtFileDir = '../dataset/Amazon/'
    f0 = open((wordsTxtFileDir + wordsTxtFileName0), 'r')
    f1 = open((wordsTxtFileDir + wordsTxtFileName1), 'r')
    remove_lf = str.maketrans('', '', '\n')
    positive_list = [x.translate(remove_lf) for x in f0]
    negative_list = [x.translate(remove_lf) for x in f1]

    # generate the list contain all the words
    all_words = []
    for t in text_processed:
        all_words.extend(t)
    all_words = list(set(all_words))  # to remove duplicate words
    all_set_words = set(all_words)  # use set to make it quicker to judge whether a word is in the corpus

    # generate the vectors of seed words
    po_list_exist = []
    ne_list_exist = []
    po_vec_exist = []
    ne_vec_exist = []
    for w in positive_list:
        if w in all_words:
            po_list_exist.append(w)
            po_vec_exist.append(model.wv[w])
    for w in negative_list:
        if w in all_words:
            ne_list_exist.append(w)
            ne_vec_exist.append(model.wv[w])
    intermediate_vb = {'po_list_exist': po_list_exist, 'ne_list_exist': ne_list_exist, 'po_vec_exist': po_vec_exist,
                       'ne_vec_exist': ne_vec_exist, 'all_words': all_words}
    # 保存intermediate_vb
    np.save('../dataset/Amazon/result/lexicon_semi/' + intermediate_vb_filename, intermediate_vb)


if __name__ == '__main__':
    word2vec_model('text_processed.npy', 'word2vec_model', 'intermediate_vb.npy')
