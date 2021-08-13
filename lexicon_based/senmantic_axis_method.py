# coding=utf-8
# import os
import json
from nltk.corpus import stopwords
import nltk.stem
import string
from gensim.models import Word2Vec
import numpy as np
import gensim
import seaborn as sns


def senmantic_axis_method(intermediate_vb_file_name, model_file_name, dic_file_name):
    # load intermediate and model
    intermediate_vb = np.load('../dataset/Amazon/result/lexicon_semi/' + intermediate_vb_file_name,
                              allow_pickle=True).item()
    po_list_exist = intermediate_vb['po_list_exist']
    ne_list_exist = intermediate_vb['ne_list_exist']
    po_vec_exist = intermediate_vb['po_vec_exist']
    ne_vec_exist = intermediate_vb['ne_vec_exist']
    all_words = intermediate_vb['all_words']
    model = gensim.models.Word2Vec.load('../dataset/Amazon/model/' + model_file_name)

    # start calculation on the paper
    V_plus = np.zeros(po_vec_exist[0].shape, dtype=float)
    V_minus = np.zeros(ne_vec_exist[0].shape, dtype=float)
    for e_w_i in po_vec_exist:
        V_plus = V_plus + e_w_i
    V_plus = V_plus / len(po_vec_exist)
    for e_w_i in ne_vec_exist:
        V_minus = V_minus + e_w_i
    V_minus = V_minus / len(ne_vec_exist)
    # calculate V_axis
    V_axis = V_plus - V_minus
    # calculate score of every word
    score_list_axis = []
    # 初始化字典
    lexicon_dic_axis = {}
    for w in all_words:
        w_vec = model.wv[w]
        cos_value = np.dot(w_vec, V_axis) / (np.linalg.norm(w_vec, ord=2) * np.linalg.norm(V_axis, ord=2))
        score_list_axis.append(cos_value)
        lexicon_dic_axis[w] = [cos_value]
    # print(score_list_axis)
    # 存储字典
    np.save('../dataset/Amazon/result/lexicon_semi/' + dic_file_name, lexicon_dic_axis)


if __name__ == '__main__':
    senmantic_axis_method('intermediate_vb.npy', 'word2vec_model', 'lexicon_dic_axis.npy')

# lexicon_dic_axis = np.load('./dataset/Amazon/result/lexicon_semi/lexicon_dic_axis.npy', allow_pickle=True).item()
# # 得到所有的值的列表
# all_values = []
# for value in lexicon_dic_axis.values():
#     all_values.append(value)
# # 打印数据的分布
# print(str(lexicon_dic_axis['bad']))
# print(str(lexicon_dic_axis['good']))
# sns.set_palette('deep', desat=.6)
# sns.set_context(rc={'figure.figsize': (8, 5)})
#
# plt.hist(np.array(all_values), bins=12, color=sns.desaturate("indianred", .8), alpha=.4)
#
# # 打印最大 和 最小
# max_value = max(all_values)
# min_value = min(all_values)
#
# max_key = []
# min_key = []
# small_key = []
# for key, value in lexicon_dic_axis.items():
#     if value == max_value:
#         max_key.append(key)
#     if value == min_value:
#         min_key.append(key)
#     if value[0] < 0:
#         small_key.append(key)
