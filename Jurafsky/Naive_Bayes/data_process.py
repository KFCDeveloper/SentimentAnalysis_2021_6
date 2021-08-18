# coding=utf-8
# @Time    : 2021/8/17 10:36
# @Author  : ydy
# @Site    : 
# @File    : data_process.py
# @Version : V 0.1
# @desc :

import numpy as np
import json

# 用来生成随机数的
np.random.randint(1000, size=20)


def result_abstract(workspace, data_name, recog_result_filename):
    """
    直接从recognition中抽取 已经识别完了的数据
    :return:
    """
    result_lexicon = np.load(workspace + recog_result_filename)
    test_set_label_ne_op = get_test_set_label_ne_op(workspace, data_name + '_' + 'test' + '_' + str(0) + '.json')
    test_set_label_ne_op = np.array(test_set_label_ne_op)
    ran_list = [486, 409, 212, 837, 314, 298, 218, 216, 37, 191, 47, 314, 568,
                634, 166, 352, 366, 856, 676, 797]
    print(result_lexicon[ran_list])
    print(test_set_label_ne_op[ran_list])


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
