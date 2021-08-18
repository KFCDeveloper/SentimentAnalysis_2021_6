# coding=utf-8
# @Time    : 2021/8/14 21:15
# @Author  : ydy
# @Site    : 
# @File    : const.py
# @Version : V 0.1
# @desc : 用来存储项目常量的

from typing import Final

# 一个review含有的句子数的最大值
MAX_NUM_SENS: Final = 40
# 一个句子含有的单词数量的最大值
MAX_NUM_WORDS: Final = 50

# 词嵌入的维度
WORD_EM_DIM: Final = 200
# 用户向量的维度
USER_DIM: Final = 200
# 商品向量的维度
PRODUCT_DIM: Final = 200

# 隐藏层 维度
HIDDEN_DIM: Final = 100
# 学习率
LEARNING_RATE: Final = 0.005

