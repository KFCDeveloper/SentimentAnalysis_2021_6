import torch
from HUAPA import HUAPA
from HUAPA_trainer import train_iters

if __name__ == '__main__':
    # 迭代次数
    n_iters = 100000
    # todo: sentence embedding 和 user product embedding 里面还没有放东西呢
    # 论文说不用dropout 各种参数都在 experiments setting中说明了
    huapa = HUAPA(sen_embeddings, user_embeddings, product_embeddings, input_dim, hid_word_dim, hid_sen_dim,
                  user_dim, product_dim, num_layers,
                  att_word_output_dim, att_sen_output_dim, classes, max_len=50, dropout=0)
    train_iters(huapa, classes, docs_classes, docs_embeddings, user_embeddings, product_embeddings, learning_rate=0.005)
    # 还差evaluation
