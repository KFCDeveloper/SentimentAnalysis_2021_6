# coding=utf-8
# import os
import json
from nltk.corpus import stopwords
import nltk.stem
import string
from gensim.models import Word2Vec
import numpy as np
import gensim
import pickle
from scipy.sparse import lil_matrix


def label_propagation(intermediate_vb_filename, word2vec_model_filename, E_filename, lexicon_dic_propagation_filename,
                      top_k):
    # load intermediate and model
    intermediate_vb = np.load('../dataset/Amazon/result/lexicon_semi/' + intermediate_vb_filename,
                              allow_pickle=True).item()
    po_list_exist = intermediate_vb['po_list_exist']
    ne_list_exist = intermediate_vb['ne_list_exist']
    po_vec_exist = intermediate_vb['po_vec_exist']
    ne_vec_exist = intermediate_vb['ne_vec_exist']
    all_words = intermediate_vb['all_words']
    model = gensim.models.Word2Vec.load('../dataset/Amazon/model/' + word2vec_model_filename)

    # initialize an empty matrix
    # it will take 47s for 10 of i. I need 18.8 days.  Time totally related to N.
    n = len(all_words)
    k = 10
    # 使用稀疏矩阵，因为我不需要切片，所以无需考虑是用 csr_matrix 还是 csc_matrix，他们两个都可以
    # 但是按照原理来说，csr_matrix 按照行来进行遍历会更快
    E = lil_matrix((n, n), dtype=float)  # define the edge
    j = 0
    for i in range(0, n):
        w_vec_i = model.wv[all_words[i]]
        z = all_words[i]
        neighbors = model.wv.most_similar(all_words[i],
                                          topn=top_k)  # 默认 k = 10, gensim还有 approximate nearest neighbor 的方法
        for neighbor in neighbors:
            # print('i:' + str(i) + ' ,j:' + str(j))
            j = all_words.index(neighbor[0])
            if j > i:  # 这样只用计算一半
                break
            similarity = model.wv.similarity(all_words[i], neighbor[0])
            E[i, j] = similarity
            E[j, i] = similarity

    # 保存 E
    E_saved = {'E': E}
    pickle.dump(E_saved, open('../dataset/Amazon/result/lexicon_semi/' + E_filename, 'wb'), protocol=4)

    E = pickle.load(open('../dataset/Amazon/result/lexicon_semi/' + E_filename, 'rb'))['E']
    # 参考维基百科 page rank，里面描述了矩阵的意义 l(p_i,p_j)是 从页面j->i的链接数/页面j中含有的外部链接总数
    # 初始化p 为 1/n
    p = np.array([1 / n] * n, dtype=float)  # 创建一个有n个元素都是1/n的list，然后再转换成 numpy (n,1)
    D = lil_matrix((n, n), dtype=float)  # 初始化D
    column_sum = E.sum(axis=0)  # 计算D
    for i in range(0, n):
        D[i, i] = column_sum[0, i]
    D_1_divide_2 = D.tocsc().sqrt()
    # 计算T
    T = D_1_divide_2.dot(E).dot(D_1_divide_2)
    # 计算s
    s_po = np.zeros([n], dtype=float)
    n_po = len(po_list_exist)
    s_ne = np.zeros([n], dtype=float)
    n_ne = len(ne_list_exist)
    for i in range(0, n):
        if all_words[i] in po_list_exist:
            s_po[i] = 1 / n_po
        if all_words[i] in ne_list_exist:
            s_ne[i] = 1 / n_ne
    # 参考的维基的阻尼系数 https://zh.wikipedia.org/zh-cn/PageRank
    beta = 0.85
    # 开始迭代   TODO: 无法确认要迭代多少次
    p_po = lil_matrix(p).transpose()
    p_ne = lil_matrix(p).transpose()
    s_po_m = lil_matrix(s_po).transpose()
    s_ne_m = lil_matrix(s_ne).transpose()
    for i in range(0, 50):
        p_po = beta * T.dot(p_po) + (1 - beta) * s_po_m
        p_ne = beta * T.dot(p_ne) + (1 - beta) * s_ne_m

    # 计算每一个单词的score
    score_plus_propagation = np.zeros([n], dtype=float)
    score_minus_propagation = np.zeros([n], dtype=float)
    # 初始化字典
    lexicon_dic_propagation = {}
    for i in range(0, n):
        if (p_po[i, 0] + p_ne[i, 0]) != 0:
            score_plus_propagation[i] = p_po[i, 0] / (p_po[i, 0] + p_ne[i, 0])
            score_minus_propagation[i] = p_ne[i, 0] / (p_po[i, 0] + p_ne[i, 0])
        # 装到字典里面，后面好取用
        lexicon_dic_propagation[all_words[i]] = [score_plus_propagation[i], score_minus_propagation[i]]
    # 存储字典
    np.save('../dataset/Amazon/result/lexicon_semi/' + lexicon_dic_propagation_filename, lexicon_dic_propagation)

    # lexicon_dic_propagation = np.load('./dataset/Amazon/result/lexicon_semi/lexicon_dic_propagation.npy',
    #                                   allow_pickle=True).item()


if __name__ == '__main__':
    label_propagation('intermediate_vb.npy', 'word2vec_model', 'E.npy', 'lexicon_dic_propagation.npy', 10)
