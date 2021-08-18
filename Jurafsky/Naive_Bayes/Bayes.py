# coding=utf-8
# @Time    : 2021/8/16 19:31
# @Author  : ydy
# @Site    : 
# @File    : Bayes.py
# @Version : V 0.1
# @desc :

import numpy as np
from data_process import get_test_set_label_ne_op


def prepare_tem(workspace, dataset_name, train_data_name):
    """
    Bayes可以提前准备好 词典和单词的频数，可以在后面预测的时候快速运算
    :return:
    """
    word_nest_list = np.load(workspace + dataset_name, allow_pickle=True).tolist()
    test_set_label_ne_op = get_test_set_label_ne_op(workspace, train_data_name + '_' + 'train' + '_' + str(0) + '.json')

    # 对于positive 和 negative 两类的 统计，并取log
    all_num = len(test_set_label_ne_op)
    po_num = np.array(test_set_label_ne_op).sum()
    ne_num = all_num - po_num
    log_p_po = np.log(po_num / all_num)
    log_p_ne = np.log(all_num - po_num / all_num)
    # 生词 辞典，词和它对应的 两个先验概率，主要要加上laplace平滑
    # dic 的形式  'word':[po_num,ne_num,log_po,log_ne]
    dic = {}
    for index, line in enumerate(word_nest_list):
        for w in line:
            if w in dic:
                if test_set_label_ne_op[index] == 1:
                    dic[w][0] += 1
                else:
                    dic[w][1] += 1
            # 单词还没有计入辞典
            else:
                if test_set_label_ne_op[index] == 1:
                    dic[w] = [1, 0, 0, 0]
                else:
                    dic[w] = [0, 1, 0, 0]

    # 统计词典中所记载的单词个数
    word_num = len(dic)
    # 遍历 dic，计算 log_po 和 log_ne
    for key, value in dic.items():
        value[2] = np.log((value[0] + 1) / (po_num + word_num))
        value[3] = np.log((value[1] + 1) / (ne_num + word_num))
    # tem 是中间值的意思
    bayes_tem = {'dic': dic, 'all_num': all_num, 'po_num': po_num, 'ne_num': ne_num, 'log_p_po': log_p_po,
                 'log_p_ne': log_p_ne}
    np.save('../../dataset/Amazon/Jurafsky_workspace/Bayes/' + 'bayes_tem.npy', bayes_tem)


def naive_bayes(sen):
    """
    传入的sentence，要判断它是positive还是negative，是一个list
    :param sen:
    :return:
    """
    bayes_tem = np.load('../../dataset/Amazon/Jurafsky_workspace/Bayes/' + 'bayes_tem.npy', allow_pickle=True).item()
    dic = bayes_tem['dic']
    # 初始化待会儿要计算的后验概率
    post_po = 0
    post_ne = 0
    for w in sen:






if __name__ == '__main__':
    workspace = '../../dataset/Amazon/lexicon_workspace/'
    data_name = 'result/lexicon_supervised/text_processed_0.npy'
    train_data_name = 'k-fold-dataset/Luxury_Beauty_5'
    prepare_tem(workspace, data_name, train_data_name)
