# coding=utf-8
# @Time    : 2021/8/17 10:36
# @Author  : ydy
# @Site    : 
# @File    : data_process.py
# @Version : V 0.1
# @desc :

import numpy as np
import json
import Bayes
import string
from nltk.corpus import stopwords
import nltk.stem

# 用来生成随机数的
# np.random.randint(1000, size=20)


def result_abstract(workspace, data_name, recog_result_filename):
    """
    直接从recognition中抽取 已经识别完了的数据
    :return:
    """
    result_lexicon = np.load(workspace + recog_result_filename)
    test_set_label_ne_op = get_test_set_label_ne_op(workspace, data_name + '_' + 'test' + '_' + str(0) + '.json')
    test_set_label_ne_op = np.array(test_set_label_ne_op)
    ran_list = [486, 409, 212, 43, 314, 298, 218, 216, 37, 77, 47, 314, 568,
                634, 166, 352, 366, 267, 676, 797]
    print(result_lexicon[ran_list])
    print(test_set_label_ne_op[ran_list])

    # 读取 测试集中对应索引的句子
    sens = read_json_by_list(ran_list)
    # 计算贝叶斯的对于抽取的句子的结果，放入 bayes_result中
    bayes_result = []
    for sen in sens:
        a_result = Bayes.naive_bayes(sen)
        bayes_result.append(a_result)
    np.save('../../dataset/Amazon/lexicon_workspace/result/lexicon_supervised/sample_result.npy',
            {'bayes_result': bayes_result, 'lexicon_result': result_lexicon[ran_list],
             'label': test_set_label_ne_op[ran_list]})
    print({'bayes_result': bayes_result, 'lexicon_result': result_lexicon[ran_list],
           'label': test_set_label_ne_op[ran_list]})


def read_json_by_list(index_list):
    """
    读取json文件中index_list对应的句子
    :return:
    """

    json_path = '../../dataset/Amazon/lexicon_workspace/k-fold-dataset/' + 'Luxury_Beauty_5_test_0.json'
    line_reviews = []
    o = open(json_path, 'r')
    for line_json in o:
        line_review = json.loads(line_json)
        line_reviews.append(line_review)
    #
    sens = []
    for index, line in enumerate(line_reviews):
        if index % 800 == 0:
            print(index)
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
            sens.append(without_stopwords)
    np.save('../../dataset/Amazon/lexicon_workspace/result/lexicon_supervised/text_processed_test_0.npy',
            {'line_reviews': sens})
    return np.array(sens)[index_list]


def get_test_set_label_ne_op(workspace, filename):
    """
    输入数据集，得到rating 小于3.1的为 negative 大于 3.1 的为正例
    :param filename:
    :return:
    """
    jsonPath = workspace + filename
    test_set_label_ne_op = []
    o = open(jsonPath, 'r')
    for line_json in open(jsonPath, 'r'):
        line = json.loads(line_json)
        if 'reviewText' in line.keys():  # to avoid json without review
            overall = line['overall']
            if overall < 3.1:
                test_set_label_ne_op.append(0)
            else:
                test_set_label_ne_op.append(1)
    return test_set_label_ne_op


if __name__ == '__main__':
    workspace = '../../dataset/Amazon/lexicon_workspace/'
    data_name = 'k-fold-dataset/Luxury_Beauty_5'
    recog_result_filename = 'result/lexicon_supervised/sentiment_recognition_log_0.npy'
    result_abstract(workspace, data_name, recog_result_filename)
