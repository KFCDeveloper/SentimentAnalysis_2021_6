# coding=utf-8
# import os
import json
from nltk.corpus import stopwords
import nltk.stem
import string
from gensim.models import Word2Vec
import numpy as np
import gensim


def sentiment_recognition(method, filename, sentiment_recognition_either_filename="",
                          sentiment_recognition_both_filename="",
                          threshold=0.0, lexicon_dic_axis_filename="", lexicon_dic_propagation_filename="",
                          pott_score_dic_filename="", log_odds_ratio_dic_filename=""):
    # 设置字典中访问value的索引
    index_both_0 = 0
    index_both_1 = 1
    index_either = 0

    # 看样子根据之前的 Semantic Axis Methods 和 Label Propagation 最后得到的结果来说，一个是 $score(w)$ 一个是 $score^+(w_i)$
    # 那应该意味着一个是要么为正要么为负，另一个是既有正值，也有负值
    if method == "axis" or method == "both":
        lexicon_dic_axis = np.load('../dataset/Amazon/result/lexicon_semi/' + lexicon_dic_axis_filename,
                                   allow_pickle=True).item()
    if method == "propagation" or method == "both":
        lexicon_dic_propagation = np.load('../dataset/Amazon/result/lexicon_semi/' + lexicon_dic_propagation_filename,
                                          allow_pickle=True).item()
    if method == "pottscore" or method == "supervised_both":
        lexicon_dic_propagation = np.load('../dataset/Amazon/result/lexicon_supervised/' + pott_score_dic_filename,
                                          allow_pickle=True).item()
        index_both_0 = 2
        index_both_1 = 3
    if method == "log_odds_ratio" or method == "supervised_both":
        lexicon_dic_axis = np.load('../dataset/Amazon/result/lexicon_supervised/' + log_odds_ratio_dic_filename,
                                   allow_pickle=True).item()
        index_either = 2
    # 加载要进行分析的语料
    jsonPath = '../dataset/Amazon/' + filename
    line_reviews = []
    o = open(jsonPath, 'r')
    for line_json in o:
        line_review = json.loads(line_json)
        line_reviews.append(line_review)

    # 定义lambda值 和 threshold
    lam = 1
    # 用来存储句子的情感，either是第一种词典得到的 要么是正要么是负数的词典 ，both是第二种词典得到的 一个单词的权重有正有负
    # 用 1 来表示 positive     0 来表示 negative
    sentiment_recognition_either = []
    sentiment_recognition_both = []
    # as for Semantic Axis Methods. Either positive or negative
    positive_score = 0
    negative_score = 0
    # as for Label Propagation. Have both positive value or negative value
    f_plus = 0
    f_minus = 0

    # begin to iteration
    # 每句话先处理，再分别计算分
    for index, line in enumerate(line_reviews):
        # 重置分数
        positive_score = 0
        negative_score = 0
        # as for Label Propagation. Have both positive value or negative value
        f_plus = 0
        f_minus = 0
        if 'reviewText' in line.keys():  # to avoid json without review
            # TODO: 下面的代码可以包装成一个 preprocess(label,language) 来进行简化
            line_review = line['reviewText']
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
            # print(index)
            # 开始计算值分析
            for w in without_stopwords:
                if 'lexicon_dic_axis' in locals().keys():  # 如果变量被定义
                    if w in lexicon_dic_axis:  # 如果训练集的这个词在词典中
                        # —————————— axis ————————————
                        w_score_0 = lexicon_dic_axis[w][index_either]
                        if w_score_0 > threshold:
                            positive_score += w_score_0
                        else:
                            # 注意，negative_score是小于0的
                            negative_score += w_score_0
                if 'lexicon_dic_propagation' in locals().keys():  # 如果变量被定义
                    if w in lexicon_dic_propagation:  # 如果训练集的这个词在词典中
                        # ——————————propagation ————————
                        w_score_1_0 = lexicon_dic_propagation[w][index_both_0]
                        w_score_1_1 = lexicon_dic_propagation[w][index_both_1]
                        f_plus += w_score_1_0
                        f_minus += w_score_1_1
        if 'lexicon_dic_axis' in locals().keys():
            # —————————— axis ————————————
            if negative_score == 0:
                sentiment_recognition_either.append(1)
            elif positive_score == 0:
                sentiment_recognition_either.append(0)
            elif abs(positive_score / negative_score) > lam:
                # print(index)
                # print(positive_score)
                # print(negative_score)
                # print(abs(positive_score / negative_score))
                sentiment_recognition_either.append(1)
            elif abs(negative_score / positive_score) > lam:
                sentiment_recognition_either.append(0)

        if 'lexicon_dic_propagation' in locals().keys():
            # ——————————propagation ————————
            if f_minus == 0:
                sentiment_recognition_both.append(1)
            elif f_plus == 0:
                sentiment_recognition_both.append(0)
            elif abs(f_plus / f_minus) > lam:
                sentiment_recognition_both.append(1)
            elif abs(f_minus / f_plus) > lam:
                sentiment_recognition_both.append(0)
    if method == "axis" or method == "both" or method == "log_odds_ratio" or method == "supervised_both":
        print('sentiment_recognition_either:' + str(sentiment_recognition_either))
        if method == "axis" or method == "both":
            np.save('../dataset/Amazon/result/lexicon_semi/' + sentiment_recognition_either_filename,
                    sentiment_recognition_either)
        elif method == "log_odds_ratio" or method == "supervised_both":
            np.save('../dataset/Amazon/result/lexicon_supervised/' + sentiment_recognition_either_filename,
                    sentiment_recognition_either)
    if method == "propagation" or method == "both" or method == "pottscore" or method == "supervised_both":
        print('sentiment_recognition_both:' + str(sentiment_recognition_both))
        if method == "propagation" or method == "both":
            np.save('../dataset/Amazon/result/lexicon_semi/' + sentiment_recognition_both_filename,
                    sentiment_recognition_both)
        elif method == "pottscore" or method == "supervised_both":
            np.save('../dataset/Amazon/result/lexicon_supervised/' + sentiment_recognition_both_filename,
                    sentiment_recognition_both)

    # sentiment_recognition_either = np.load('./dataset/Amazon/result/lexicon_semi/sentiment_recognition_either.npy')
    # sentiment_recognition_both = np.load('./dataset/Amazon/result/lexicon_semi/sentiment_recognition_both.npy')
    # print(sentiment_recognition_both)


if __name__ == '__main__':
    # sentiment_recognition(lexicon_dic_axis_filename='k-fold/lexicon_dic_axis' + '_' + str(4) + '.npy',
    #                       lexicon_dic_propagation_filename='k-fold/lexicon_dic_propagation' + '_' + str(4) + '.npy',
    #                       filename='k-fold-dataset/Luxury_Beauty_5' + '_' + 'test' + '_' + str(
    #                           4) + '.json',
    #                       sentiment_recognition_either_filename='k-fold/sentiment_recognition_either' + '_' + str(
    #                           4) + '.npy',
    #                       sentiment_recognition_both_filename='k-fold/sentiment_recognition_both' + '_' + str(
    #                           4) + '.npy', method="both")

    # sentiment_recognition('lexicon_dic_axis.npy', 'lexicon_dic_propagation.npy', 'Luxury_Beauty_5.json',
    #                       'sentiment_recognition_either.npy', 'sentiment_recognition_both.npy')
    # sentiment_recognition(method='supervised_both', threshold=0, pott_score_dic_filename='lexicon_dic_pottscore.npy',
    #                       log_odds_ratio_dic_filename='lexicon_dic_log.npy', filename='Luxury_Beauty_5.json',
    #                       sentiment_recognition_either_filename='sentiment_recognition_log.npy',
    #                       sentiment_recognition_both_filename='sentiment_recognition_pott.npy')

    # sentiment_recognition(method='supervised_both', threshold=0,
    #                       pott_score_dic_filename='k-fold/lexicon_dic_pottscore' + '_' + str(
    #                           0) + '.npy',
    #                       log_odds_ratio_dic_filename='k-fold/lexicon_dic_log' + '_' + str(
    #                           0) + '.npy',
    #                       filename='k-fold-dataset/Luxury_Beauty_5' + '_' + 'test' + '_' + str(
    #                           0) + '.json',
    #                       sentiment_recognition_either_filename='k-fold/sentiment_recognition_log' + '_' + str(
    #                           0) + '.npy',
    #                       sentiment_recognition_both_filename='k-fold/sentiment_recognition_pott' + '_' + str(
    #                           0) + '.npy')

    sentiment_recognition(method='log_odds_ratio', threshold=0.3225,
                          log_odds_ratio_dic_filename='k-fold/lexicon_dic_log' + '_' + str(
                              0) + '.npy',
                          filename='k-fold-dataset/Luxury_Beauty_5' + '_' + 'test' + '_' + str(
                              0) + '.json',
                          sentiment_recognition_either_filename='k-fold/sentiment_recognition_log' + '_' + str(
                              0) + '.npy')

