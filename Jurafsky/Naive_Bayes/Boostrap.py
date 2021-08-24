# coding=utf-8
# @Time    : 2021/8/16 21:08
# @Author  : ydy
# @Site    : 
# @File    : Boostrap.py
# @Version : V 0.1
# @desc :   贝叶斯分类的实现

import numpy as np
import random


# 先算了 delta_x 能够得知 log_ratio的性能好于 bayes，那么在大样本下，也是如此么？
# 那么先假设 log_ratio是比bayes差的，
# 采样出 b 个样本，计算性能差，发现在大部分的样本上面，log_ratio是比bayes好，那么就reject，得出log_ratio是好于bayes

def boostrap(b):
    """
    use boostrap to evaluate the difference of performance between A and B
    :param b:   num of samples b
    :param A:
    :param B:
    :return:    p-values
    """
    result_label = np.load('../../dataset/Amazon/lexicon_workspace/result/lexicon_supervised/sample_result.npy',
                           allow_pickle=True).item()
    bayes_result = result_label['bayes_result']
    lexicon_result = result_label['lexicon_result']
    label = result_label['label']
    delta_x = measure(lexicon_result, bayes_result, label)
    s = 0  # 对应algorithm中的0
    # 开始进行 sample
    for i in range(b):
        rand_list = [random.randint(0, 19) for j in range(b)]
        delta_x_b = measure(np.array(lexicon_result)[rand_list], np.array(bayes_result)[rand_list],
                            np.array(label)[rand_list])
        if delta_x_b > delta_x * 2:
            s += 1
    return s / b


def measure(result_list_a, result_list_b, label):
    acu_a = 0
    acu_b = 0
    for i in range(len(result_list_a)):
        if result_list_a[i] == label[i]:
            acu_a += 1
        if result_list_b[i] == label[i]:
            acu_b += 1
    return (acu_a - acu_b) / len(result_list_a)


if __name__ == '__main__':
    workspace = '../../dataset/Amazon/lexicon_workspace/'
    data_name = 'result/lexicon_supervised/text_processed_0.npy'
    print('%.11f' % boostrap(200))
