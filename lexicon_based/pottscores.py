# coding=utf-8
# import os
import json
import string

import nltk.stem
from nltk.corpus import stopwords
from lexicon_based.preprocess import sentence_process
import numpy as np


def pott_score(file_name, pott_score_dic_filename):
    # 我们定义1~3是negative  4~5 是positive
    # 初始化一个字典，每个key对应的value都会是一个list，
    # list[0]表示rating1~3的出现的次数，list[1]表示rating4~5出现的次数，
    # list[2]表示p_w_c_negative的值，list[3]表示p_w_c_positive的值
    pott_score_dic = {}
    # 统计每个单词的出现次数

    # read the corpus json file
    # do preprocessing
    # 路径要改，因为python console的当前路径是 /data/student2020/ydy/SentimentAnalysis_2021_6
    json_path = '../dataset/Amazon/' + file_name
    o = open(json_path, 'r')
    for index, line_json in enumerate(open(json_path, 'r')):
        # print(index)
        line = json.loads(line_json)
        if 'reviewText' in line.keys():  # to avoid json without review
            line_review = line['reviewText']
            rating = line['overall']
            # 对句子进行处理一些例行处理
            without_stopwords = sentence_process(line_review)
            # 遍历这句话的每一个单词，进行统计
            for w in without_stopwords:
                if w not in pott_score_dic:
                    pott_score_dic[w] = []
                    if rating < 3.1:
                        pott_score_dic[w].append(1)
                        pott_score_dic[w].append(0)
                    else:
                        pott_score_dic[w].append(0)
                        pott_score_dic[w].append(1)
                else:
                    if rating < 3.1:
                        pott_score_dic[w][0] += 1
                    else:
                        pott_score_dic[w][1] += 1

    # 统计两个corpus分别有多少个单词
    count_rating_1_3 = 0
    count_rating_4_5 = 0
    for key, value in pott_score_dic.items():
        count_rating_1_3 += value[0]
        count_rating_4_5 += value[1]
    # 计算pott score
    for key, value in pott_score_dic.items():
        pott_score_dic[key].append(value[0] / count_rating_1_3)
        pott_score_dic[key].append(value[1] / count_rating_4_5)
    np.save('../dataset/Amazon/result/lexicon_supervised/' + pott_score_dic_filename, pott_score_dic)


if __name__ == '__main__':

    # pott_score('Luxury_Beauty_5.json', 'lexicon_dic_pottscore.npy')
    pott_score_dic = np.load('../dataset/Amazon/result/lexicon_supervised/' + 'lexicon_dic_pottscore.npy',
                                 allow_pickle=True).item()
    print(1)
