# coding=utf-8
# @Time    : 2021/8/13 10:38
# @Author  : ydy
# @Site    : 
# @File    : embedding_vec.py
# @Version : V 0.1
# @desc : 主要是用 transformer 生成单词的embedding，然后装入Embedding中；
# 然后生成 user 和 product 的embedding，或者说是代表它们id的唯一向量，
# 然后看是否要装入Embedding中


import numpy as np
import transformers as ppb


def gen_word_em():
    # For DistilBERT:
    model_class, tokenizer_class, pretrained_weights = (
        ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    distilbert_model = model_class.from_pretrained(pretrained_weights)


if __name__ == '__main__':
    gen_word_em()
