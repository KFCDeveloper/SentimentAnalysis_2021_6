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
import pandas as pd
import transformers as ppb
import torch
from const import (MAX_NUM_SENS, MAX_NUM_WORDS)


def gen_word_em(csv_workspace, raw_name):
    """

    :param workspace:
    :param raw:
    :return:
    """
    ''' # 加载DistilBERT模型，以及它的权重'''
    # For DistilBERT:
    model_class, tokenizer_class, pretrained_weights = (
        ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    DistilBERT_model = model_class.from_pretrained(pretrained_weights)

    ''' # 处理数据格式成array，生成mask'''
    # 读入数据，转换成numpy格式，发现速度很快，1个G的数据转换非常迅速

    # 这里 我的数据中的句子是缩成一团的，是一个嵌套的list
    # 首先我可以确认的是，我肯定不能 for循环输入 pre-trained model，for的效率超级低，所以我将 第三列嵌套的给展开，
    # 主要是要记住它们的索引，属于哪一句，然后再堆叠起来
    big_sens_array, index_array = generate_array(csv_workspace, raw_name)
    # 生成 mask_array
    attention_mask = np.where(big_sens_array != '\\space', 1, 0)
    # 转换成tensor，好输入进模型
    input_ids = torch.tensor(big_sens_array)
    attention_mask = torch.tensor(attention_mask)

    ''' # 输入数据，生成词嵌入'''
    # 输入 pre-trained model
    with torch.no_grad():
        last_hidden_states = DistilBERT_model(input_ids, attention_mask=attention_mask)


def generate_array(csv_workspace, raw_name):
    """
    用来输入原始的dataframe，将嵌套的list全部展开，方便输入 pre-trained 模型
    :param csv_workspace:
    :param raw_name:
    :return:
    """
    df = pd.read_csv(csv_workspace + raw_name + '_padding.csv')

    sen_nest_df = df['reviewText']
    (rows,) = sen_nest_df.shape

    unfold_series = pd.DataFrame(columns=[str(i) for i in range(50)])
    index_series = pd.DataFrame(columns=['paragraph_index'])
    # 开始遍历嵌套list，这里我要看一下速度，要是过慢，就保存一下
    series_index = 0
    for row in range(rows):
        para = eval(sen_nest_df[row])
        for i in range(len(para)):
            # 在空的dataframe中添加行
            index_series.loc[series_index] = [row]
            unfold_series.loc[series_index] = para[i]
            series_index += 1

        if row % 1000 == 0:
            print(row)

    # 访问大的数据，转化成numpy好像pycharm会很卡，我选择dataframe   ❌转化成numpy
    unfold_series.to_csv(csv_workspace + 'tempo_data/' + raw_name + '_unfold_series.csv', index=False)
    index_series.to_csv(csv_workspace + 'tempo_data/' + raw_name + '_index_series.csv', index=False)
    return unfold_series, index_series


if __name__ == '__main__':
    csv_workspace = './dataset/Amazon/huapa_workspace/data_processed/'
    raw_name = 'Video_Games_5'
    generate_array(csv_workspace, raw_name)
    # gen_word_em(csv_workspace, raw_name)
