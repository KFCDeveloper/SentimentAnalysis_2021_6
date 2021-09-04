# coding=utf-8
# @Time    : 2021/8/19 16:05
# @Author  : ydy
# @Site    : 
# @File    : data_process.py
# @Version : V 0.1
# @desc :  之前的数据处理全部白干， 人家是输入预训练模型的就是一组一组的句子，并且 要用别人的dictionary才行
# 我自己处理的，再进行索引，简直多此一举，前面也不是白干，至少如果以后遇到没有直接用的工具，我还能够有数据处理的思路的

import numpy as np
import pandas as pd
import transformers as ppb
import torch
import os
from math import ceil


def generate_embedding(csv_workspace, raw_name):
    """
    生成 100000个句子的embedding
    :param csv_workspace:
    :param raw_name:
    :return:
    """
    # 很多都是参照了博客 https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
    # For DistilBERT:
    model_class, tokenizer_class, pretrained_weights = (
        ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    DistilBERT_model = model_class.from_pretrained(pretrained_weights)

    # 判断是否已经计算出了 临时文件 - 被分词且被索引的document
    if not os.path.exists(csv_workspace + 'tempo_data/' + raw_name + '_tem_tokenized.npy'):
        # 加载展开的句子 list 和 index_list
        tem_result = np.load(csv_workspace + raw_name + '_tem_unfold_.npy', allow_pickle=True).item()
        # for i in range(len(tem_result['unfold_list'])):
        #     a = tokenizer.encode(tem_result['unfold_list'][i], add_special_tokens=True)
        #     if len(a) > 50:
        #         print('wrong:'+str(i))
        #     if i % 100000 == 0:
        #         print(i)
        tokenized = [tokenizer.encode(sen, add_special_tokens=True) for sen in tem_result['unfold_list']]
        np.save(csv_workspace + 'tempo_data/' + raw_name + '_tem_tokenized.npy', {'tokenized': tokenized})
    else:
        tokenized = np.load(csv_workspace + 'tempo_data/' + raw_name + '_tem_tokenized.npy', allow_pickle=True).item()[
            'tokenized']
    # padding
    max_len = 50
    # 清除其中为 100 的单词，代表词库没有，应该属于无效的单词
    tokenized = [list(filter((100).__ne__, sen)) for sen in tokenized]
    # 截取50个，因为首位有 DistilBERT 生成的符号，所以保留48个      (list要进行修改必须使用索引)
    for index in range(len(tokenized)):
        if len(tokenized[index]) > 50:
            tokenized[index] = tokenized[index][:49] + [tokenized[index][-1]]
    # 进行padding (md别人的程序写的就这么好，我的就要写for循环，臭臭)
    padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized])
    # 生成 mask 矩阵
    attention_mask = np.where(padded != 0, 1, 0)
    # 使用 tensor 包裹
    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)

    # 生成 embedding
    # 这里直接转换这么大的矩阵，好像爆内存了，就一部分一部分转换吧
    # 现在发现就这样也不行，last_hidden_states 太大了，不行的
    # last_hidden_states = []
    em_size = 1000  # 用来表示一次性输入进模型的size
    total_size = len(tokenized)
    # em_batch_nums = int(total_size / em_size) + (0 if total_size % em_size == 0 else 1)  # 记录分批输入的批次
    # 由于完整得输出 330万 条句子，将会使得数据集达到123GB，
    # 所以，我只取10万条了，大概3GB左右吧，但是也是使用的分批次喂入模型，防止全部读入内存造成资源占用过大
    em_batch_nums = 100
    with torch.no_grad():
        for batch_now in range(99, em_batch_nums):
            if batch_now != em_batch_nums - 1:  # 还没到  最后一个可能不为完整的 em_size 的batch
                last_hidden_states = DistilBERT_model(input_ids[batch_now * em_size:batch_now * em_size + em_size],
                                                      attention_mask=attention_mask[batch_now * em_size:
                                                                                    batch_now * em_size + em_size])
            else:  # 到达了最后一个 batch
                last_hidden_states = DistilBERT_model(input_ids[batch_now * em_size:batch_now * em_size + em_size],
                                                      attention_mask=attention_mask[
                                                                     batch_now * em_size:batch_now * em_size + em_size])
            np.save(
                csv_workspace + 'embedding_fold/' + raw_name + '_word_embedding_last_hidden_' + str(batch_now) + '.npy',
                {'last_hidden': last_hidden_states})
            print(batch_now)


# DistilBERT_model(torch.tensor(np.array([[101,300,400,102],[101,300,400,102],[101,300,400,102]])))
def generate_nest_list(csv_workspace, raw_name):
    """
    通过dataframe的 'reviewText' 属性里面的值生成嵌套的list和对应的index_list
    :param df:
    :return:
    """
    # 读入数据
    df = pd.read_csv(csv_workspace + raw_name + '_cleaned.csv')
    # 得到嵌套数据的行数
    sen_nest_df = df['reviewText']
    (rows,) = sen_nest_df.shape

    # 统计 嵌套数据展开后的行数
    rows_expand = 0
    for row in range(rows):
        para = eval(sen_nest_df[row])
        rows_expand += len(para)
        if row % 10000 == 0:
            print(row)
    # 先初始化一个numpy的数组，然后再转成dataframe，如果是直接在dataframe上面进行增删是非常耗费资源的
    # 因为dataframe的内存是一块连续区域，要增加，只能整个复制到一块新的连续内存区域中，所以非常耗费资源
    # 我在尝试了 numpy 修改数组后发现，也很慢，还是就老实用列表好了

    index_list = []
    unfold_list = []
    # 开始遍历嵌套list，这里我要看一下速度，要是过慢，就保存一下
    series_index = 0
    for row in range(rows):
        para = eval(sen_nest_df[row])
        for i in range(len(para)):
            sen = para[i]
            if not sen:  # 过滤掉分词阶段出现的 []
                continue
            if i == 40:  # 限制最大句子数
                break
            # 在空的dataframe中添加行
            index_list += [row]
            unfold_list.append(sen)
            series_index += 1

        if row % 10000 == 0:
            print(row)
    tem_result = {'index_list': index_list, 'unfold_list': unfold_list}
    np.save(csv_workspace + 'tempo_data/' + raw_name + '_tem_unfold_.npy', tem_result)


def clean_dataset(csv_workspace, raw_name):
    """
    清理数据，除去带有没有意义的空值的行
    :param csv_workspace:
    :param raw_name:
    :return:
    """
    df = pd.read_csv(csv_workspace + raw_name + '.csv')
    # 去除 verified  reviewTime  reviewerName  summary  unixReviewTime  vote  style  image  这些列
    # 只保留 overall  reviewerID  asin  reviewText
    df = df[['overall', 'reviewerID', 'asin', 'reviewText']]
    # 去掉有 Nah 的行
    df = df.dropna(axis=0, how='any')
    # 去掉为 [[]] 的行
    df = df[~df['reviewText'].isin(['[[]]'])]
    df.to_csv(csv_workspace + raw_name + '_cleaned.csv', index=False)


def product_em(csv_workspace, raw_name):
    """
    生成product的embeddings
    :return:
    """
    # 读入前10w行的 product 的 id
    df = pd.read_csv(csv_workspace + raw_name + '_cleaned.csv')
    product_frame = df['asin']
    product_dic = {}
    for index in range(len(product_frame)):
        if product_frame[index] not in product_dic:
            product_dic[product_frame[index]] = np.random.uniform(-0.01, 0.01, 200)
    np.save(csv_workspace + 'tempo_data/' + raw_name + '_product_dic.npy', {'product_dic': product_dic})


def user_em(csv_workspace, raw_name):
    df = pd.read_csv(csv_workspace + raw_name + '_cleaned.csv')
    user_frame = df['asin']
    user_dic = {}
    for index in range(len(user_frame)):
        if user_frame[index] not in user_dic:
            user_dic[user_frame[index]] = np.random.uniform(-0.01, 0.01, 200)

    np.save(csv_workspace + 'tempo_data/' + raw_name + '_user_dic.npy', {'user_dic': user_dic})


if __name__ == '__main__':
    csv_workspace = '../dataset/Amazon/huapa_workspace/data_processed/'
    raw_name = 'Video_Games_5'
    # clean_dataset(csv_workspace, raw_name)
    a = np.load(csv_workspace + 'tempo_data/' + raw_name + '_product_dic.npy', allow_pickle=True).item()['product_dic']
    # user_em(csv_workspace, raw_name)
    # generate_id_dataframe(csv_workspace, raw_name)
    # print(1)
