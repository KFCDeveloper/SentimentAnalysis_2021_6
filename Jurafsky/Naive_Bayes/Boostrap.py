# coding=utf-8
# @Time    : 2021/8/16 21:08
# @Author  : ydy
# @Site    : 
# @File    : Boostrap.py
# @Version : V 0.1
# @desc :   贝叶斯分类的实现

import numpy as np


def boostrap(x, b, A, B):
    """
    use boostrap to evaluate the difference of performance between A and B
    :param x:   test set x
    :param b:   num of samples b
    :param A:
    :param B:
    :return:    p-values
    """


def load_data(workspace, data_name):
    """
    按道理是不应该在这里写数据加载的，不过为了方便罢了，我也不想写到 data_process里去了
    将json转化成numpy
    :param workspace:
    :param data_name:
    :return:
    """
    print(1)


if __name__ == '__main__':
    workspace = '../../dataset/Amazon/lexicon_workspace/'
    data_name = 'result/lexicon_supervised/text_processed_0.npy'
    load_data(workspace, data_name)
