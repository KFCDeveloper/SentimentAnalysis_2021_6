import torch
from HUAPA import HUAPA
from HUAPA_trainer import train_iters

if __name__ == '__main__':
    # 迭代次数
    n_iters = 100000
    # todo: sentence embedding 和 user product embedding 里面还没有放东西呢




    # 论文说不用dropout 各种参数都在 experiments setting中说明了

    train_iters(classes, docs_classes, docs_embeddings, user_embeddings, product_embeddings, learning_rate=0.005)
    # 还差evaluation
