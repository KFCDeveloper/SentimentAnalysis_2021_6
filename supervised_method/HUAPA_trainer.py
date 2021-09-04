import torch
import torch.nn as nn
import torch.nn.functional as f
import BiLSTMAttn
from torch import optim
import HUAPA
import numpy as np


def train(model: HUAPA.HUAPA, optimizer, classes, docs_classes, docs_embeddings, user_embeddings, product_embeddings):
    """

    :param model:
    :param optimizer:
    :param classes:
    :param docs_classes:
    :param docs_embeddings:
    :param user_embeddings:
    :param product_embeddings:
    :return:
    """
    # 做完一次反向传播过后，肯定要让梯度归零
    optimizer.zero_grad()
    d_user, d_pro, d, p, p_u, p_p = model(docs_embeddings, user_embeddings, product_embeddings)
    ''' # 计算loss'''
    # todo:不能确定这个不是Parameter的能不能被自动微分
    loss_1, loss_2, loss_3, L = torch.nn.Parameter(torch.zeros(1, dtype=torch.double))
    lambda_1, lambda_2, lambda_3 = torch.nn.Parameter(torch.randn(1, dtype=torch.double))
    # 需要的是 一个doc的类别和它的索引
    for i, doc_c in enumerate(docs_classes):
        for j, c in enumerate(classes):
            loss_1 += (-1) * (1 if c == doc_c else 0) * np.log(p[j])
            loss_2 += (-1) * (1 if c == doc_c else 0) * np.log(p_u[j])
            loss_3 += (-1) * (1 if c == doc_c else 0) * np.log(p_p[j])
    L = lambda_1 * loss_1 + lambda_2 * loss_2 + lambda_3 * loss_3
    L.backward()
    optimizer.step()
    return L


def train_iters(classes, docs_classes, docs_embeddings, user_embeddings, product_embeddings,
                n_iters, learning_rate=0.005):
    """

    :param classes:
    :param docs_classes:
    :param docs_embeddings:
    :param user_embeddings:
    :param product_embeddings:
    :param n_iters: 迭代的次数
    :param learning_rate:
    :return:
    """
    huapa = HUAPA.HUAPA(sen_embeddings, user_embeddings, product_embeddings, input_dim, hid_word_dim, hid_sen_dim,
                  user_dim, product_dim, num_layers,
                  att_word_output_dim, att_sen_output_dim, classes, max_len=50, dropout=0)
    adam = optim.Adam(huapa.parameters(), lr=learning_rate)
    for iter in range(1, n_iters + 1):
        loss = train(huapa, adam, classes, docs_classes, docs_embeddings, user_embeddings, product_embeddings)
        # 收集 loss 保存模型之类的  做一些保存收集数据的动作

    # 所有的训练都结束了，在这里画图，画出过程
