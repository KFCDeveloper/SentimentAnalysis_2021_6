# coding=utf-8
# import os
import json
from nltk.corpus import stopwords
import nltk.stem
import string
from gensim.models import Word2Vec
import numpy as np
import gensim

# extract the review to train on word2vec
# and I find word2vec only receive list of lists
# https://towardsdatascience.com/a-beginners-guide-to-word-embedding-with-gensim-word2vec-model-5970fa56cc92
# nltk.download('punkt')   # english word tokenize
# nltk.download('stopwords')

# read the corpus json file
# do preprocessing
# 路径要改，因为python console的当前路径是 /data/student2020/ydy/SentimentAnalysis_2021_6
fileName = 'Luxury_Beauty_5.json'
jsonPath = './dataset/Amazon/' + fileName
text_processed = []
o = open(jsonPath, 'r')
for line_json in open(jsonPath, 'r'):
    line_review = json.loads(line_json)
    if 'reviewText' in line_review.keys():  # to avoid json without review
        line_review = line_review['reviewText']
        # preprocessing  https://blog.csdn.net/weixin_43216017/article/details/88324093
        # transfer to lower case
        line_review_lower = line_review.lower()
        # remove the punctuation
        remove = str.maketrans('', '', string.punctuation)
        without_punctuation = line_review_lower.translate(remove)
        # tokenize
        tokens = nltk.word_tokenize(without_punctuation)
        # remove the word not used
        without_stopwords = [w for w in tokens if w not in stopwords.words('english')]
        # to extract the stem  ## I find this will transfer nothing to noth
        # s = nltk.stem.SnowballStemmer('english')  # para is the chosen language
        # cleaned_text = [s.stem(ws) for ws in without_stopwords]
        text_processed.append(without_stopwords)
# print(text_processed)

# 保存text_processed，即预处理后的嵌套列表
text_processed = np.array(text_processed, dtype=list)
np.save('./dataset/Amazon/result/lexicon_semi/text_processed.npy', text_processed)

# https://blog.csdn.net/u010665216/article/details/78709018 sentence generator
# get the list of positive and negative words

# 加载预处理后的数据
text_processed = np.load('./dataset/Amazon/result/lexicon_semi/text_processed.npy',
                         allow_pickle=True).tolist()
# 1. 训练模型 并进行存储，之前非常疑惑输入进word2vec的sentences是什么，现在由
# [博客](https://towardsdatascience.com/a-beginners-guide-to-word-embedding-with-gensim-word2vec-model-5970fa56cc92)
# 可以知道，输入进入word2vec的sentences就是嵌套的`list`
# 2. 存储word2vec，以及使用sentence generator（当数据过大的时候，需要使用`yield`来慢慢生成sentence）
# 参考 [博客](https://blog.csdn.net/u010665216/article/details/78709018)
# train the model
model = Word2Vec(text_processed, min_count=1, vector_size=50, workers=3, window=3, sg=1)
# 存储模型
model.save('./dataset/Amazon/model/' + 'word2vec_model')

# generate the list of seed words
wordsTxtFileName0 = 'positive-words.txt'
wordsTxtFileName1 = 'negative-words.txt'
wordsTxtFileDir = './dataset/Amazon/'
f0 = open((wordsTxtFileDir + wordsTxtFileName0), 'r')
f1 = open((wordsTxtFileDir + wordsTxtFileName1), 'r')
remove_lf = str.maketrans('', '', '\n')
positive_list = [x.translate(remove_lf) for x in f0]
negative_list = [x.translate(remove_lf) for x in f1]

# generate the list contain all the words
all_words = []
for t in text_processed:
    all_words.extend(t)
all_words = list(set(all_words))  # to remove duplicate words
all_set_words = set(all_words)  # use set to make it quicker to judge whether a word is in the corpus

# generate the vectors of seed words
po_list_exist = []
ne_list_exist = []
po_vec_exist = []
ne_vec_exist = []
for w in positive_list:
    if w in all_words:
        po_list_exist.append(w)
        po_vec_exist.append(model.wv[w])
for w in negative_list:
    if w in all_words:
        ne_list_exist.append(w)
        ne_vec_exist.append(model.wv[w])
intermediate_vb = {'po_list_exist': po_list_exist, 'ne_list_exist': ne_list_exist, 'po_vec_exist': po_vec_exist,
                   'ne_vec_exist': ne_vec_exist, 'all_words': all_words}
# 保存intermediate_vb
np.save('./dataset/Amazon/result/lexicon_semi/intermediate_vb.npy', intermediate_vb)

# ————————————————————————————————————————————— Semantic Axis Methods ——————————————————————————————————————————————————
# ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# load intermediate and model
intermediate_vb = np.load('./dataset/Amazon/result/lexicon_semi/intermediate_vb.npy',
                          allow_pickle=True).item()
po_list_exist = intermediate_vb['po_list_exist']
ne_list_exist = intermediate_vb['ne_list_exist']
po_vec_exist = intermediate_vb['po_vec_exist']
ne_vec_exist = intermediate_vb['ne_vec_exist']
all_words = intermediate_vb['all_words']
model = gensim.models.Word2Vec.load('./dataset/Amazon/model/' + 'word2vec_model')

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
np.save('./dataset/Amazon/result/lexicon_semi/lexicon_dic_axis.npy', lexicon_dic_axis)

# ———————————————————————————————————————————————————— Label Propagation ———————————————————————————————————————————————
# ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# initialize an empty matrix
# it will take 47s for 10 of i. I need 18.8 days.  Time totally related to N.
n = len(all_words)
E = np.zeros([n, n], dtype=float)  # define the edge
for i in range(0, n):
    w_vec_i = model.wv[all_words[i]]
    for j in range(0, n):
        w_vec_j = model.wv[all_words[j]]
        v = (-1) * np.dot(w_vec_i, w_vec_j) / (np.linalg.norm(w_vec_i, ord=2) * np.linalg.norm(w_vec_j, ord=2))
        print('i:' + str(i) + ',j:' + str(j) + '\n')
        # 为了解决精度问题
        if v > 1:
            v = 1
        elif v < -1:
            v = -1
        E[i][j] = np.arccos(v)
# 参考维基百科 page rank，里面描述了矩阵的意义 l(p_i,p_j)是 从页面j->i的链接数/页面j中含有的外部链接总数
# 初始化p 为 1/n
p = np.array([1 / n] * n, dtype=float)  # 创建一个有n个元素都是1/n的list，然后再转换成 numpy
D = np.zeros([n, n], dtype=float)  # 初始化D
column_sum = E.sum(axis=0)  # 计算D
for i in range(0, n):
    D[i][i] = column_sum[i]
D_1_divide_2 = np.sqrt(D)
# 计算T
T = D_1_divide_2 * E * D_1_divide_2
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
p_po = p
p_ne = p
for i in range(0, 50):
    p_po = beta * np.dot(T, p_po) + (1 - beta) * s_po
    p_ne = beta * np.dot(T, p_ne) + (1 - beta) * s_ne

# 计算每一个单词的score
score_plus_propagation = np.zeros([n], dtype=float)
score_minus_propagation = np.zeros([n], dtype=float)
# 初始化字典
lexicon_dic_propagation = {}
for i in range(0, n):
    score_plus_propagation[i] = p_po[i] / (p_po[i] + p_ne[i])
    score_minus_propagation[i] = p_ne[i] / (p_po[i] + p_ne[i])
    # 装到字典里面，后面好取用
    lexicon_dic_propagation[all_words[i]] = [score_plus_propagation[i], score_minus_propagation[i]]
# 存储字典
np.save('./dataset/Amazon/result/lexicon_semi/lexicon_dic_propagation.npy', lexicon_dic_propagation)

# —————————————————————— Using Lexicons for Sentiment Recognition ——————————————————————————————————————————————————————
# ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# 看样子根据之前的 Semantic Axis Methods 和 Label Propagation 最后得到的结果来说，一个是 $score(w)$ 一个是 $score^+(w_i)$
# 那应该意味着一个是要么为正要么为负，另一个是既有正值，也有负值

lexicon_dic_axis = np.load('./dataset/Amazon/result/lexicon_semi/lexicon_dic_axis.npy', allow_pickle=True).item()
# lexicon_dic_propagation = np.load('./dataset/Amazon/result/lexicon_semi/lexicon_dic_propagation.npy',
#                                   allow_pickle=True).item()
# 加载要进行分析的语料
fileName = 'Luxury_Beauty_5.json'
jsonPath = './dataset/Amazon/' + fileName
line_reviews = []
o = open(jsonPath, 'r')
for line_json in o:
    line_review = json.loads(line_json)
    line_reviews.append(line_review)

# 定义lambda值
lam = 1
# 用来存储句子的情感，either是第一种词典得到的 要么是正要么是负数的词典 ，both是第二种词典得到的 一个单词的权重有正有负
# 用 1 来表示 positive     0 来表示 negative
sentiment_recognition_either = []
sentiment_recognition_both = []
# as for Semantic Axis Methods. Either positive or negative
positive_score = 0
negative_score = 0
# as for Label Propagation. Have both positive value or negative value
f_plus = 0
f_minus = 0

# begin to iteration
# 每句话先处理，再分别计算分
for index, line in enumerate(line_reviews):
    if 'reviewText' in line.keys():  # to avoid json without review
        # TODO: 下面的代码可以包装成一个 preprocess(label,language) 来进行简化
        line_review = line['reviewText']
        # preprocessing  https://blog.csdn.net/weixin_43216017/article/details/88324093
        # transfer to lower case
        line_review_lower = line_review.lower()
        # remove the punctuation
        remove = str.maketrans('', '', string.punctuation)
        without_punctuation = line_review_lower.translate(remove)
        # tokenize
        tokens = nltk.word_tokenize(without_punctuation)
        # remove the word not used
        without_stopwords = [w for w in tokens if w not in stopwords.words('english')]
        # to extract the stem  ## I find this will transfer nothing to noth
        # s = nltk.stem.SnowballStemmer('english')  # para is the chosen language
        # cleaned_text = [s.stem(ws) for ws in without_stopwords]

        # 开始计算值分析
        for w in without_stopwords:
            # —————————— axis ————————————
            # if w=='handcream':
            #     continue
            w_score_0 = lexicon_dic_axis[w][0]
            if w_score_0 > 0:
                positive_score += w_score_0
            else:
                # 注意，negative_score是小于0的
                negative_score += w_score_0

            # # ——————————propagation ————————
            # w_score_1_0 = lexicon_dic_propagation[w][0]
            # w_score_1_1 = lexicon_dic_propagation[w][1]
            # f_plus += w_score_1_0
            # f_minus += w_score_1_1

    # —————————— axis ————————————
    if negative_score == 0:
        sentiment_recognition_either.append(1)
    elif positive_score == 0:
        sentiment_recognition_either.append(0)
    elif abs(positive_score / negative_score) > lam:
        sentiment_recognition_either.append(1)
    elif abs(negative_score / positive_score) > lam:
        sentiment_recognition_either.append(0)

    # # ——————————propagation ————————
    # if f_minus == 0:
    #     sentiment_recognition_both.append(1)
    # elif f_plus == 0:
    #     sentiment_recognition_both.append(0)
    # elif abs(f_plus / f_minus) > lam:
    #     sentiment_recognition_both.append(1)
    # elif abs(f_minus / f_plus) > lam:
    #     sentiment_recognition_both.append(0)
print('sentiment_recognition_either:'+str(sentiment_recognition_either))
print('sentiment_recognition_both:'+str(sentiment_recognition_both))
np.save('./dataset/Amazon/result/lexicon_semi/sentiment_recognition_either.npy',
        sentiment_recognition_either)
np.save('./dataset/Amazon/result/lexicon_semi/sentiment_recognition_both.npy',
        sentiment_recognition_both)
