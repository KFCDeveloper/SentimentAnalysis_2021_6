# coding=utf-8
# import os
import json
from nltk.corpus import stopwords
import nltk.stem
import string
from gensim.models import Word2Vec
import numpy as np
import re
import gensim


# extract the review to train on word2vec
# and I find word2vec only receive list of lists
# https://towardsdatascience.com/a-beginners-guide-to-word-embedding-with-gensim-word2vec-model-5970fa56cc92
# nltk.download('punkt')   # english word tokenize
# nltk.download('stopwords')

def preprocess_semi(fileName, savedName):
    # read the corpus json file
    # do preprocessing
    # 路径要改，因为python console的当前路径是 /data/student2020/ydy/SentimentAnalysis_2021_6
    jsonPath = '../dataset/Amazon/' + fileName
    text_processed = []
    o = open(jsonPath, 'r')
    for line_json in open(jsonPath, 'r'):
        line = json.loads(line_json)
        if 'reviewText' in line.keys():  # to avoid json without review
            line_review = line['reviewText']
            without_stopwords = sentence_process(line_review)
            text_processed.append(without_stopwords)
    # print(text_processed)

    # 保存text_processed，即预处理后的嵌套列表
    text_processed = np.array(text_processed, dtype=list)
    np.save('../dataset/Amazon/result/lexicon_semi/' + savedName, text_processed)

    # https://blog.csdn.net/u010665216/article/details/78709018 sentence generator
    # get the list of positive and negative words


def sentence_process(line_review):
    # preprocessing  https://blog.csdn.net/weixin_43216017/article/details/88324093
    # transfer to lower case
    line_review_lower = line_review.lower()
    # remove the punctuation
    remove = str.maketrans('', '', string.punctuation)
    without_punctuation = line_review_lower.translate(remove)
    # tokenize
    tokens = nltk.word_tokenize(without_punctuation)
    # remove the word not used
    without_stopwords = [w for w in tokens if w not in stopwords.words('english')]
    # to extract the stem  ## I find this will transfer nothing to noth
    # s = nltk.stem.SnowballStemmer('english')  # para is the chosen language
    # cleaned_text = [s.stem(ws) for ws in without_stopwords]

    # 剔除含有数字的
    pattern = re.compile('[0-9]+')
    cleaned_text = [w for w in without_stopwords if not pattern.findall(w)]
    return cleaned_text


if __name__ == '__main__':
    preprocess_semi('Luxury_Beauty_5.json', 'text_processed.npy')
