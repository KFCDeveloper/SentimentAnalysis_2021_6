# coding=utf-8
# import os
import json
import string

import nltk.stem
from nltk.corpus import stopwords
from lexicon_based.preprocess import sentence_process
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# 得到一个单词在corpus i,j 中出现的次数，然后在background corpus 中出现的次数，以及background corpus的大小
def log_odds_ratio(file_name, log_odds_ratio_dic_filename, log_intermediate_filename, log_intermediate_bgc_filename):
    """
    # 得到一个单词在corpus i,j 中出现的次数，然后在background corpus 中出现的次数，以及background corpus的大小
    :return:
    """
    # 我们定义1~3是negative  4~5 是positive
    # 初始化一个字典，每个key对应的value都会是一个list，
    # list[0]表示rating1~3的出现的次数，list[1]表示rating4~5出现的次数，
    # list[2]表示value
    log_odds_ratio_dic = {}
    # 统计每个单词的出现次数

    # read the corpus json file
    # do preprocessing
    # 路径要改，因为python console的当前路径是 /data/student2020/ydy/SentimentAnalysis_2021_6
    json_path = '../dataset/Amazon/' + file_name
    o = open(json_path, 'r')
    for line_json in open(json_path, 'r'):
        line = json.loads(line_json)
        if 'reviewText' in line.keys():  # to avoid json without review
            line_review = line['reviewText']
            rating = line['overall']
            # 对句子进行处理一些例行处理
            without_stopwords = sentence_process(line_review)
            # 遍历这句话的每一个单词，进行统计
            for w in without_stopwords:
                if w not in log_odds_ratio_dic:
                    log_odds_ratio_dic[w] = []
                    # 由paper中的公式可以看出，当\delta的值越大的时候，word更可能出现在corpus i 中，
                    # 所以，corpus i是正例才能满足\delta越大，越positive，这是我弄错的原因
                    if rating < 3.1:
                        log_odds_ratio_dic[w].append(0)
                        log_odds_ratio_dic[w].append(1)
                    else:
                        log_odds_ratio_dic[w].append(1)
                        log_odds_ratio_dic[w].append(0)
                else:
                    if rating < 3.1:
                        log_odds_ratio_dic[w][1] += 1
                    else:
                        log_odds_ratio_dic[w][0] += 1

    log_intermediate = {'log_odds_ratio_dic': log_odds_ratio_dic}
    np.save('../dataset/Amazon/result/lexicon_supervised/' + log_intermediate_filename, log_intermediate)
    # todo: ./ 改成 ../
    log_intermediate = np.load('../dataset/Amazon/result/lexicon_supervised/' + log_intermediate_filename,
                               allow_pickle=True).item()
    log_intermediate_bgc = np.load('../dataset/Amazon/result/lexicon_supervised/' + log_intermediate_bgc_filename,
                                   allow_pickle=True).item()
    log_odds_ratio_dic = log_intermediate['log_odds_ratio_dic']
    bgc_word_ids = log_intermediate_bgc['bgc_word_ids']
    wiki_lexicon_w2i = log_intermediate_bgc['wiki_lexicon_w2i']
    wiki_lexicon_i2w = log_intermediate_bgc['wiki_lexicon_i2w']
    # 遍历每一个单词，计算z_score，bgc是background corpus
    # 在字典中的value中加上这个单词在语料中出现的次数
    for w_id in bgc_word_ids:
        w = wiki_lexicon_i2w[w_id][0]
        wiki_lexicon_w2i[w][1] += 1
    # bg语料的大小
    alpha_0 = len(bgc_word_ids)

    # 计算 n_i 和 n_j 的值
    # 统计两个corpus分别有多少个单词
    n_i = 0
    n_j = 0
    for key, value in log_odds_ratio_dic.items():
        n_i += value[0]
        n_j += value[1]
    # print(1)
    # 开始计算每一个单词的delta，还有z-score
    for key, value in log_odds_ratio_dic.items():
        # print(key)
        f_w_i = log_odds_ratio_dic[key][0]
        f_w_j = log_odds_ratio_dic[key][1]
        alpha_w = 0
        if key in wiki_lexicon_w2i:
            alpha_w = wiki_lexicon_w2i[key][1]
        if (n_j + alpha_0 - f_w_j - alpha_w) == 0:
            break
        delta_w_i_j = np.log((f_w_i + alpha_w + 1) / (n_i + alpha_0 - f_w_i - alpha_w + 1)) - np.log(
            (f_w_j + alpha_w + 1) / (n_j + alpha_0 - f_w_j - alpha_w + 1))

        sigma_2 = 1 / (f_w_i + alpha_w + 1) + 1 / (f_w_j + alpha_w + 1)
        z_score = delta_w_i_j / np.sqrt(sigma_2)
        log_odds_ratio_dic[key].append(z_score)
    np.save('../dataset/Amazon/result/lexicon_supervised/' + log_odds_ratio_dic_filename, log_odds_ratio_dic)


def deal_bgc(log_intermediate_bgc_filename):
    bgc_text_ids, bgc_seq_ids, bgc_word_ids = get_bgc()
    wiki_lexicon_w2i, wiki_lexicon_i2w = get_bgc_lexicon()
    log_intermediate_bgc = {'bgc_text_ids': bgc_text_ids,
                            'bgc_seq_ids': bgc_seq_ids, 'bgc_word_ids': bgc_word_ids,
                            'wiki_lexicon_w2i': wiki_lexicon_w2i, 'wiki_lexicon_i2w': wiki_lexicon_i2w}
    np.save('../dataset/Amazon/result/lexicon_supervised/' + log_intermediate_bgc_filename, log_intermediate_bgc)


def get_bgc():
    """
    bgc : background corpus
    :return:
    """

    wiki_dataset_file_path = '../dataset/Wiki/' + 'database.txt'
    text_ids = []
    seq_ids = []
    word_ids = []
    # 遍历 行和行索引
    for index, line in enumerate(open(wiki_dataset_file_path, 'r')):
        # 分词
        tokens = nltk.word_tokenize(line)
        # 去掉前面的空白无用内容
        if index > 2:
            text_ids.append(tokens[0])
            seq_ids.append(tokens[1])
            word_ids.append(tokens[2])
    return text_ids, seq_ids, word_ids


def get_bgc_lexicon():
    wiki_lexicon_file_path = '../dataset/Wiki/' + 'wiki-samples-lexicon.txt'
    # 得到一个  word->index  以及 index->word的字典
    wiki_lexicon_w2i = {}
    wiki_lexicon_i2w = {}
    # 遍历 行和行索引
    for index, line in enumerate(open(wiki_lexicon_file_path, 'r')):
        # 分词
        tokens = nltk.word_tokenize(line)
        # 去掉前面的空白无用内容
        if index > 2:
            # a = tokens[1]
            # if a in wiki_lexicon_w2i:
            #     print(1)
            wiki_lexicon_w2i[tokens[1]] = []
            wiki_lexicon_w2i[tokens[1]].append(tokens[0])
            wiki_lexicon_w2i[tokens[1]].append(0)  # 表示这个单词在语料中出现的次数
            wiki_lexicon_i2w[tokens[0]] = []
            wiki_lexicon_i2w[tokens[0]].append(tokens[1])
    return wiki_lexicon_w2i, wiki_lexicon_i2w


def inspection():
    """
    检查并确认最大和最小值
    :return:
    """
    log_odds_ratio_dic = np.load('../dataset/Amazon/result/lexicon_supervised/' + 'lexicon_dic_log.npy',
                                 allow_pickle=True).item()
    max = 0
    min = 0
    for key, value in log_odds_ratio_dic.items():
        if value[2] > max and value[2] != float("inf"):
            max = value[2]
        if value[2] < min and value[2] != float("-inf"):
            min = value[2]
    print(max)  # 25.38580291973807         设置 inf = 30
    print(min)  # -27.8825629746497   所以我决定设置 -inf = -30


def post_process(log_odds_ratio_dic_processed_filename):
    """
    将正负无穷替换掉
    :return:
    """
    log_odds_ratio_dic = np.load('../dataset/Amazon/result/lexicon_supervised/' + 'lexicon_dic_log.npy',
                                 allow_pickle=True).item()
    for key, value in log_odds_ratio_dic.items():
        if value[2] == float("inf"):
            value[2] = 30
        if value[2] == float("-inf"):
            value[2] = -30
    np.save('../dataset/Amazon/result/lexicon_supervised/' + log_odds_ratio_dic_processed_filename, log_odds_ratio_dic)


def draw():
    sns.set()
    log_odds_ratio_dic = np.load('../dataset/Amazon/result/lexicon_supervised/' + 'lexicon_dic_log.npy',
                                 allow_pickle=True).item()
    value_list = []
    value_list_below_0 = []
    value_list_above_0 = []
    for index, value in log_odds_ratio_dic.items():
        value_list.append(value[2])
    value_list.sort()
    for i in range(len(value_list)):
        if i == int(len(value_list) / 2):
            print(value_list[i])


def export():
    """
    排序后导出成txt
    :return:
    """
    log_odds_ratio_dic = np.load('../dataset/Amazon/result/lexicon_supervised/' + 'lexicon_dic_log.npy',
                                 allow_pickle=True).item()
    list_result = []
    for key, value in log_odds_ratio_dic.items():
        item = [key, value[2]]
        list_result.append(item)
    list_result.sort(key=lambda x: x[1])
    with open(
            '../dataset/Amazon/result/lexicon_supervised/' + 'lexicon_dic_log_sort_result.txt',
            'w') as f_target:
        for data in list_result:
            f_target.write(str(data)+'\n')

    with open(
            '../dataset/Amazon/result/lexicon_supervised/' + 'lexicon_dic_log_sort_result_negative.txt',
            'w') as f_target:
        for i in range(200):
            f_target.write(str(list_result[i])+'\n')

    n = len(list_result)
    with open(
            '../dataset/Amazon/result/lexicon_supervised/' + 'lexicon_dic_log_sort_result_positive.txt',
            'w') as f_target:
        for i in range(200):
            f_target.write(str(list_result[n-i-1])+'\n')


if __name__ == '__main__':
    # get_bgc_lexicon()
    # get_bgc()

    # log_odds_ratio('Luxury_Beauty_5.json', 'lexicon_dic_log.npy', 'log_intermediate.npy')
    # log_intermediate_filename = 'k-fold/log_intermediate_filename_0.npy'
    # log_odds_ratio_dic_filename = 'k-fold/lexicon_dic_log_0.npy'
    # post_process('lexicon_dic_log_processed.npy')

    # log_odds_ratio_dic = np.load('./dataset/Amazon/result/lexicon_supervised/' + 'log_intermediate.npy',
    #                              allow_pickle=True).item()
    # deal_bgc('log_intermediate_gbc.npy')
    # log_odds_ratio('Luxury_Beauty_5.json', 'lexicon_dic_log.npy', 'log_intermediate.npy', 'log_intermediate_gbc.npy')
    # draw()

    export()
