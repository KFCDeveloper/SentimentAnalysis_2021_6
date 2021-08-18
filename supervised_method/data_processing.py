import json
import re
import string
import pandas as pd

import nltk
import numpy as np
from nltk.corpus import stopwords
# 这里python常数的管理参考博客 https://github.com/home-assistant/core/blob/dev/homeassistant/const.py
from const import (MAX_NUM_SENS, MAX_NUM_WORDS)


# df = data = pd.DataFrame(np.arange(16).reshape((4, 4)), index=['Ohio', 'Colorado', 'Utah', 'New York'],
#                          columns=['one', 'two', 'three', 'four'])
# df = df[~df['one'].isin(['[[]]'])]


def transfer_to_csv(file_path, workspace, file_name):
    """
    将经过reformat的json文件转化成csv
    :param file_path: reformat后的json文件的位置
    :param workspace: 工作空间
    :param file_name: 数据集本来的名字，要用来命名csv文件的名字
    :return:
    """
    df = pd.read_json(file_path, encoding="utf-8", orient='records')
    csv_name = file_name[:-5] + '.csv'
    df.to_csv(workspace + '/data_processed/' + csv_name, index=False)


def reformat(workspace, file_name):
    """
    将中间没有逗号连接的 json文件中加上逗号，同时在首位加上 []，这样成为一整个json文件
    :param workspace: 工作空间位置
    :param file_name: 输入的格式不满足 json_read() 的文件名
    :return:
    """
    json_path = workspace + file_name
    re_file_name = file_name[:-5] + '_reformat' + '.json'
    ob_list = []
    for index, line_json in enumerate(open(json_path, 'r')):
        line = json.loads(line_json)
        if 'reviewText' in line.keys():  # to avoid json without review
            line_review = line['reviewText']
            without_stopwords = paragraph_process(line_review)
            line['reviewText'] = without_stopwords
        ob_list.append(line)
        if index % 30 == 0:
            print(index)
    json_array = json.dumps(ob_list, ensure_ascii=False)
    file_path = workspace + '/data_processed/' + re_file_name
    with open(file_path, 'w') as f_target:
        f_target.write(json_array)
    return file_path


def paragraph_process(line_review):
    """
    对输入的review text进行分句 每句进行分词，然后返回嵌套列表
    :param line_review:
    :return:
    """
    # preprocessing  https://blog.csdn.net/weixin_43216017/article/details/88324093
    # transfer to lower case
    line_review_lower = line_review.lower()
    # 先分句，每个句子再进行分词
    sen_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    # 对句子进行分割
    sentences = sen_tokenizer.tokenize(line_review_lower)
    sen_list = []
    for sen in sentences:
        # remove the punctuation
        remove = str.maketrans('', '', string.punctuation)
        without_punctuation = sen.translate(remove)
        # tokenize
        tokens = nltk.word_tokenize(without_punctuation)
        # remove the word not used
        without_stopwords = [w for w in tokens if w not in stopwords.words('english')]
        # to extract the stem  ## I find this will transfer nothing to noth
        # s = nltk.stem.SnowballStemmer('english')  # para is the chosen language
        # cleaned_text = [s.stem(ws) for ws in without_stopwords]
        # 剔除含有数字的
        pattern = re.compile('[0-9]+')
        cleaned_text = [w for w in without_stopwords if not pattern.findall(w)]
        sen_list.append(cleaned_text)
    return sen_list


def clean_dataset(csv_workspace, raw_name):
    df = pd.read_csv(csv_workspace + raw_name + '_cleaned.csv')
    # 去除 verified  reviewTime  reviewerName  summary  unixReviewTime  vote  style  image  这些列
    # 只保留 overall  reviewerID  asin  reviewText
    df = df[['overall', 'reviewerID', 'asin', 'reviewText']]
    # 去掉有 Nah 的行
    df = df.dropna(axis=0, how='any')
    # 去掉为 [[]] 的行
    df = df[~df['reviewText'].isin(['[[]]'])]
    df.to_csv(csv_workspace + raw_name + '_cleaned.csv', index=False)


def padding(csv_workspace, raw_name):
    df = pd.read_csv(csv_workspace + raw_name + '_cleaned.csv')
    # for 循环非常消耗计算资源  这篇比较非常棒 https://zhuanlan.zhihu.com/p/97269320
    df_len = len(df)
    for i in range(len(df)):
        paragraph = eval(df.iat[i, 3])
        paras_len = len(paragraph)

        j = 0
        for _ in range(paras_len):
            if j == MAX_NUM_SENS:
                break
            sen_len = len(paragraph[j])
            paragraph[j] = paragraph[j] + ['\\space' for k in range(MAX_NUM_WORDS - sen_len)]
            paragraph[j] = paragraph[j][:MAX_NUM_WORDS]
            j += 1

        df.iat[i, 3] = paragraph[:j]
        # if i % 1000 == 0:
        #     print(i)
    df.to_csv(csv_workspace + raw_name + '_padding.csv', index=False)


if __name__ == '__main__':
    # 格式化 和 转换文件的格式
    # dataset_workspace = './dataset/Amazon/huapa_workspace/'
    # dataset_name = 'Video_Games_5.json'
    # data_re_path = reformat(dataset_workspace, dataset_name)
    # transfer_to_csv(data_re_path, dataset_workspace, dataset_name)

    # 清洗空白的行 和  不需要的列， 我们只需要 product_id  user_id  review_text
    # csv_workspace = '../dataset/Amazon/huapa_workspace/data_processed/'
    # raw = 'Video_Games_5'
    # clean_dataset(csv_workspace, raw)

    # 将句子对齐，paper中说，句子数量最多40，一句的单词数量最大不超过50，max length 就是50
    csv_workspace = '../dataset/Amazon/huapa_workspace/data_processed/'
    raw = 'Video_Games_5'
    padding(csv_workspace, raw)

    # 截取 Video_Games_5_padding.csv 的前1000行作为 test_padding.csv,
    # 因为跑完整的数据展开的时候太慢了，我先用1000行来debug应该够了
    df = pd.read_csv(csv_workspace + raw + '_padding.csv')
    df[:1000].to_csv(csv_workspace + 'test' + '_padding.csv', index=False)
