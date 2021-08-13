# coding=utf-8
# import os
import json

import numpy as np

from lexicon_based import label_propagation
from lexicon_based import preprocess
from lexicon_based import senmantic_axis_method
from lexicon_based import sentiment_recognition
from lexicon_based import word2vec_model


def train_test_split_lexicon(k):
    jsonPath = '../dataset/Amazon/' + 'Luxury_Beauty_5.json'

    # 存放数据
    datalist = []
    o = open(jsonPath, 'r')
    for line_json in open(jsonPath, 'r'):
        datalist.append(line_json)

    n = len(datalist)
    stride = int(n / k) + 1  # 一份数据的大小
    # 编号
    flag = 0
    data_lists_save = []
    data_list_save = []
    # 生成测试集
    for i in range(0, n):
        data_list_save.append(datalist[i])
        if (i + 1) % stride == 0 or i == n - 1:
            with open(
                    '../dataset/Amazon/k-fold-dataset/' + 'Luxury_Beauty_5' + '_' + 'test' + '_' + str(flag) + '.json',
                    'w') as f_target:
                for data in data_list_save:
                    f_target.write(data)
            data_lists_save.append(data_list_save)
            data_list_save = []
            flag += 1
    # 生成训练集
    for i in range(0, len(data_lists_save)):
        data_list_train_save = []
        # 生成1：4的训练集
        for j in range(0, len(data_lists_save)):
            if i != j:
                data_list_train_save += data_lists_save[j]
        with open(
                '../dataset/Amazon/k-fold-dataset/' + 'Luxury_Beauty_5' + '_' + 'train' + '_' + str(i) + '.json',
                'w') as f_target:
            for data in data_list_train_save:
                f_target.write(data)
        # 每一个train dataset 可以对应生成四个 1：3的训练集和验证集
        for k in range(0, len(data_lists_save)):
            vali_train_dataset_save = []
            if k != i:
                # 保存验证集
                with open(
                        '../dataset/Amazon/k-fold-dataset/validation/' + 'Luxury_Beauty_5' + '_' + 'train' + '_' + str(
                            i) + '_' + 'vali' + '_' + str(k) + '.json',
                        'w') as f_target:
                    for data in data_lists_save[k]:
                        f_target.write(data)
                # 开始生成 1:3 训练集
                for p in range(0, len(data_lists_save)):
                    if p != k and p != i:
                        vali_train_dataset_save += data_lists_save[p]
                with open(
                        '../dataset/Amazon/k-fold-dataset/validation/' + 'Luxury_Beauty_5' + '_' + 'train' + '_' + str(
                            i) + '_' + 'train' + '_' + str(k) + '.json',
                        'w') as f_target:
                    for data in vali_train_dataset_save:
                        f_target.write(data)
            # 生成验证集 将上面4份的训练集分成 3份的训练集 和 1份的验证集


def validation_dataset_processed(k):
    """
    先对于 验证的 1:3训练集和验证集先进行处理 保存中间量
    :param k:
    :return:
    """
    # 先把数据进行预处理
    for i in range(k):
        for j in range(k):
            if j != i:
                print('i:' + str(i) + ' j:' + str(j) + ' begin')
                preprocess.preprocess_semi(
                    'k-fold-dataset/validation/Luxury_Beauty_5' + '_' + 'train' + '_' + str(
                        i) + '_' + 'train' + '_' + str(j) + '.json',
                    'k-fold/validation/text_processed' + '_' + str(i) + '_' + 'vali' + '_' + str(j) + '.npy')
                print('i:' + str(i) + ' j:' + str(j) + ' begin0')
                word2vec_model.word2vec_model(
                    'k-fold/validation/text_processed' + '_' + str(i) + '_' + 'vali' + '_' + str(j) + '.npy',
                    'k-fold/validation/word2vec_model' + '_' + str(i) + '_' + 'vali' + '_' + str(j),
                    'k-fold/validation/intermediate_vb' + '_' + str(i) + '_' + 'vali' + '_' + str(j) + '.npy')
                print('i:' + str(i) + ' j:' + str(j) + ' begin1')


def validation_axis(k):
    """
    使用validation dataset来选择合适的 k（top-k in matrix E）默认使用的 10
    从训练集当中再分出1份出来当作验证集
    :return:

    首先测试 [10, 100, 1000, 10000]，确定大体的位置
    """
    TP_eithers = []
    FN_eithers = []
    FP_eithers = []
    TN_eithers = []

    Acurracy_eithers = []
    Recall_eithers = []
    Precision_eithers = []
    F1_measure_eithers = []
    grid_threshold = [0.3, 0.4, 0.5, 0.6]

    for threshold in grid_threshold:
        for i in range(k):
            for j in range(k):
                if j != i:
                    print('i:' + str(i) + ' j:' + str(j) + ' begin2')
                    senmantic_axis_method.senmantic_axis_method(
                        'k-fold/validation/intermediate_vb' + '_' + str(i) + '_' + 'vali' + '_' + str(j) + '.npy',
                        'k-fold/validation/word2vec_model' + '_' + str(i) + '_' + 'vali' + '_' + str(j),
                        'k-fold/validation/lexicon_dic_axis' + '_' + str(i) + '_' + 'vali' + '_' + str(j) + '.npy')
                    print('i:' + str(i) + ' j:' + str(j) + ' begin3')
                    sentiment_recognition.sentiment_recognition(method="axis", threshold=threshold,
                                                                lexicon_dic_axis_filename='k-fold/validation/lexicon_dic_axis' + '_' + str(
                                                                    i) + '_' + 'vali' + '_' + str(j) + '.npy',
                                                                filename='k-fold-dataset/validation/Luxury_Beauty_5' + '_' + 'train' + '_' + str(
                                                                    i) + '_' + 'vali' + '_' + str(j) + '.json',
                                                                sentiment_recognition_either_filename='k-fold/validation/sentiment_recognition_either' + '_' + str(
                                                                    i) + '_' + 'vali' + '_' + str(j) + '.npy')
                    print('i:' + str(i) + ' j:' + str(j) + ' begin4')
                    # 计算accuracy
                    sentiment_recognition_either = np.load(
                        '../dataset/Amazon/result/lexicon_semi/k-fold/validation/sentiment_recognition_either' + '_' + str(
                            i) + '_' + 'vali' + '_' + str(j) + '.npy')
                    test_set_label_ne_op = get_test_set_label_ne_op(
                        'k-fold-dataset/validation/Luxury_Beauty_5' + '_' + 'train' + '_' + str(
                            i) + '_' + 'vali' + '_' + str(
                            j) + '.json')
                    eva_calculate_general(sentiment_recognition_either, test_set_label_ne_op, TP_eithers, FN_eithers,
                                          FP_eithers,
                                          TN_eithers, Acurracy_eithers, Recall_eithers, Precision_eithers,
                                          F1_measure_eithers)
    result = {'TP_eithers': TP_eithers, 'FN_eithers': FN_eithers,
              'FP_eithers': FP_eithers, 'TN_eithers': TN_eithers,
              'Acurracy_eithers': Acurracy_eithers,
              'Recall_eithers': Recall_eithers,
              'Precision_eithers': Precision_eithers,
              'F1_measure_eithers': F1_measure_eithers}
    # 保存intermediate_vb
    np.save('../dataset/Amazon/result/lexicon_semi/k-fold/validation/' + '1_' + 'vali_axis_result.npy',
            result)


def validation_propagation(k):
    TP_boths = []
    FN_boths = []
    FP_boths = []
    TN_boths = []

    Acurracy_boths = []
    Recall_boths = []
    Precision_boths = []
    F1_measure_boths = []
    grid_top_k = [10, 100, 1000, 10000]
    # 手动模拟网格搜索 propagation 方法的 top_k
    for top_k in grid_top_k:
        for i in range(k):
            for j in range(k):
                if j != i:
                    print('i:' + str(i) + ' j:' + str(j) + ' begin2')
                    label_propagation.label_propagation(
                        'k-fold/validation/intermediate_vb' + '_' + str(i) + '_' + 'vali' + '_' + str(j) + '.npy',
                        'k-fold/validation/word2vec_model' + '_' + str(i) + '_' + 'vali' + '_' + str(j),
                        'k-fold/validation/E' + '_' + str(i) + '_' + 'vali' + '_' + str(j) + '.npy',
                        'k-fold/validation/lexicon_dic_propagation' + '_' + str(
                            i) + '_' + 'vali' + '_' + str(j) + '.npy',
                        top_k=top_k)
                    print('i:' + str(i) + ' j:' + str(j) + ' begin3')
                    sentiment_recognition.sentiment_recognition(
                        method="propagation",
                        lexicon_dic_propagation_filename='k-fold/validation/lexicon_dic_propagation' + '_' + str(
                            i) + '_' + 'vali' + '_' + str(j) + '.npy',
                        filename='k-fold-dataset/validation/Luxury_Beauty_5' + '_' + 'train' + '_' + str(
                            i) + '_' + 'vali' + '_' + str(j) + '.json',
                        sentiment_recognition_both_filename='k-fold/validation/sentiment_recognition_both' + '_' + str(
                            i) + '_' + 'vali' + '_' + str(j) + '.npy')
                    print('i:' + str(i) + ' j:' + str(j) + ' begin4')
                    # 计算accuracy
                    sentiment_recognition_both = np.load(
                        '../dataset/Amazon/result/lexicon_semi/k-fold/validation/sentiment_recognition_both' + '_' + str(
                            i) + '_' + 'vali' + '_' + str(j) + '.npy')
                    test_set_label_ne_op = get_test_set_label_ne_op(
                        'k-fold-dataset/validation/Luxury_Beauty_5' + '_' + 'train' + '_' + str(
                            i) + '_' + 'vali' + '_' + str(
                            j) + '.json')
                    eva_calculate_general(sentiment_recognition_both, test_set_label_ne_op, TP_boths, FN_boths,
                                          FP_boths, TN_boths,
                                          Acurracy_boths, Recall_boths, Precision_boths, F1_measure_boths)
    result = {'TP_boths': TP_boths,
              'FN_boths': FN_boths,
              'FP_boths': FP_boths,
              'TN_boths': TN_boths,
              'Acurracy_boths': Acurracy_boths,
              'Recall_boths': Recall_boths,
              'Precision_boths': Precision_boths,
              'F1_measure_boths': F1_measure_boths}
    # 保存intermediate_vb
    np.save('../dataset/Amazon/result/lexicon_semi/k-fold/validation/' + '1_' + 'vali_propagation_result.npy', result)


def evaluation_semi(k):
    TP_eithers = []
    TP_boths = []
    FN_eithers = []
    FN_boths = []
    FP_eithers = []
    FP_boths = []
    TN_eithers = []
    TN_boths = []

    Acurracy_eithers = []
    Acurracy_boths = []
    Recall_eithers = []
    Recall_boths = []
    Precision_eithers = []
    Precision_boths = []
    F1_measure_eithers = []
    F1_measure_boths = []

    for flag in range(0, k):
        print(str(flag) + 'begin')
        preprocess.preprocess_semi('k-fold-dataset/Luxury_Beauty_5' + '_' + 'train' + '_' + str(flag) + '.json',
                                   'k-fold/text_processed' + '_' + str(flag) + '.npy')
        print(str(flag) + 'begin0')
        word2vec_model.word2vec_model('k-fold/text_processed' + '_' + str(flag) + '.npy',
                                      'k-fold/word2vec_model' + '_' + str(flag),
                                      'k-fold/intermediate_vb' + '_' + str(flag) + '.npy')
        print(str(flag) + 'begin1')
        senmantic_axis_method.senmantic_axis_method('k-fold/intermediate_vb' + '_' + str(flag) + '.npy',
                                                    'k-fold/word2vec_model' + '_' + str(flag),
                                                    'k-fold/lexicon_dic_axis' + '_' + str(flag) + '.npy')
        print(str(flag) + 'begin2')
        label_propagation.label_propagation('k-fold/intermediate_vb' + '_' + str(flag) + '.npy',
                                            'k-fold/word2vec_model' + '_' + str(flag),
                                            'k-fold/E' + '_' + str(flag) + '.npy',
                                            'k-fold/lexicon_dic_propagation' + '_' + str(flag) + '.npy', top_k=10)
        print(str(flag) + 'begin3')

        sentiment_recognition.sentiment_recognition(method="both", threshold=0.4,
                                                    lexicon_dic_axis_filename='k-fold/lexicon_dic_axis' + '_' + str(
                                                        flag) + '.npy',
                                                    lexicon_dic_propagation_filename='k-fold/lexicon_dic_propagation' + '_' + str(
                                                        flag) + '.npy',
                                                    filename='k-fold-dataset/Luxury_Beauty_5' + '_' + 'test' + '_' + str(
                                                        flag) + '.json',
                                                    sentiment_recognition_either_filename='k-fold/sentiment_recognition_either' + '_' + str(
                                                        flag) + '.npy',
                                                    sentiment_recognition_both_filename='k-fold/sentiment_recognition_both' + '_' + str(
                                                        flag) + '.npy')
        # 计算accuracy
        sentiment_recognition_either = np.load(
            '../dataset/Amazon/result/lexicon_semi/k-fold/sentiment_recognition_either' + '_' + str(
                flag) + '.npy')
        sentiment_recognition_both = np.load(
            '../dataset/Amazon/result/lexicon_semi/k-fold/sentiment_recognition_both' + '_' + str(
                flag) + '.npy')
        test_set_label_ne_op = get_test_set_label_ne_op('k-fold-dataset/Luxury_Beauty_5' + '_' + 'test' + '_' + str(
            flag) + '.json')
        eva_calculate_general(sentiment_recognition_either, test_set_label_ne_op, TP_eithers, FN_eithers, FP_eithers,
                              TN_eithers, Acurracy_eithers, Recall_eithers, Precision_eithers, F1_measure_eithers)
        eva_calculate_general(sentiment_recognition_both, test_set_label_ne_op, TP_boths, FN_boths, FP_boths, TN_boths,
                              Acurracy_boths, Recall_boths, Precision_boths, F1_measure_boths)
    result = {'TP_eithers': TP_eithers, 'TP_boths': TP_boths, 'FN_eithers': FN_eithers, 'FN_boths': FN_boths,
              'FP_eithers': FP_eithers, 'FP_boths': FP_boths, 'TN_eithers': TN_eithers, 'TN_boths': TN_boths,
              'Acurracy_eithers': Acurracy_eithers, 'Acurracy_boths': Acurracy_boths, 'Recall_eithers': Recall_eithers,
              'Recall_boths': Recall_boths,
              'Precision_eithers': Precision_eithers, 'Precision_boths': Precision_boths,
              'F1_measure_eithers': F1_measure_eithers, 'F1_measure_boths': F1_measure_boths}
    # 保存intermediate_vb
    np.save('../dataset/Amazon/result/lexicon_semi/k-fold/' + 'result.npy', result)


def eva_calculate_general(sentiment_recognition_result, test_set_label_ne_op, TPs, FNs, FPs, TNs, Accuracys, Recalls,
                          Precisions, F1_measures):
    TP = 0
    FN = 0
    FP = 0
    TN = 0

    for i in range(0, min(len(sentiment_recognition_result), len(test_set_label_ne_op))):
        if sentiment_recognition_result[i] == 1 and test_set_label_ne_op[i] == 1:
            TP += 1
        elif sentiment_recognition_result[i] == 0 and test_set_label_ne_op[i] == 1:
            FN += 1
        elif sentiment_recognition_result[i] == 1 and test_set_label_ne_op[i] == 0:
            FP += 1
        elif sentiment_recognition_result[i] == 0 and test_set_label_ne_op[i] == 0:
            TN += 1
    Acurracy = get_accuracy(TP, FN, FP, TN)
    Recall = get_recall(TP, FN, FP, TN)
    Precision = get_precision(TP, FN, FP, TN)
    F1_measure = get_f1_measure(1, Precision, Recall)

    TPs.append(TP)
    FNs.append(FN)
    FPs.append(FP)
    TNs.append(TN)

    Accuracys.append(Acurracy)
    Recalls.append(Recall)
    Precisions.append(Precision)
    F1_measures.append(F1_measure)


# this method has been abandoned
def eva_calculate(sentiment_recognition_either, sentiment_recognition_both, test_set_label_ne_op, TP_eithers,
                  FN_eithers,
                  FP_eithers, TN_eithers, TP_boths, FN_boths, FP_boths, TN_boths, Acurracy_eithers, Acurracy_boths,
                  Recall_eithers, Recall_boths, Precision_eithers, Precision_boths, F1_measure_eithers,
                  F1_measure_boths):
    TP_either = 0
    FN_either = 0
    TP_both = 0
    FN_both = 0
    FP_either = 0
    FP_both = 0
    TN_either = 0
    TN_both = 0

    for i in range(0, min(len(sentiment_recognition_either), len(test_set_label_ne_op),
                          len(sentiment_recognition_both))):
        if sentiment_recognition_either[i] == 1 and test_set_label_ne_op[i] == 1:
            TP_either += 1
        elif sentiment_recognition_either[i] == 0 and test_set_label_ne_op[i] == 1:
            FN_either += 1
        elif sentiment_recognition_either[i] == 1 and test_set_label_ne_op[i] == 0:
            FP_either += 1
        elif sentiment_recognition_either[i] == 0 and test_set_label_ne_op[i] == 0:
            TN_either += 1

        if sentiment_recognition_both[i] == 1 and test_set_label_ne_op[i] == 1:
            TP_both += 1
        elif sentiment_recognition_both[i] == 0 and test_set_label_ne_op[i] == 1:
            FN_both += 1
        elif sentiment_recognition_both[i] == 1 and test_set_label_ne_op[i] == 0:
            FP_both += 1
        elif sentiment_recognition_both[i] == 0 and test_set_label_ne_op[i] == 0:
            TN_both += 1
    Acurracy_either = get_accuracy(TP_either, FN_either, FP_either, TN_either)
    Acurracy_both = get_accuracy(TP_both, FN_both, FP_both, TN_both)
    Recall_either = get_recall(TP_either, FN_either, FP_either, TN_either)
    Recall_both = get_recall(TP_both, FN_both, FP_both, TN_both)
    Precision_either = get_precision(TP_either, FN_either, FP_either, TN_either)
    Precision_both = get_precision(TP_both, FN_both, FP_both, TN_both)
    F1_measure_either = get_f1_measure(1, Precision_either, Recall_either)
    F1_measure_both = get_f1_measure(1, Precision_both, Recall_both)

    TP_eithers.append(TP_either)
    FN_eithers.append(FN_either)
    FP_eithers.append(FP_either)
    TN_eithers.append(TN_either)

    TP_boths.append(TP_both)
    FN_boths.append(FN_both)
    FP_boths.append(FP_both)
    TN_boths.append(TN_both)

    Acurracy_eithers.append(Acurracy_either)
    Acurracy_boths.append(Acurracy_both)
    Recall_eithers.append(Recall_either)
    Recall_boths.append(Recall_both)
    Precision_eithers.append(Precision_either)
    Precision_boths.append(Precision_both)
    F1_measure_eithers.append(F1_measure_either)
    F1_measure_boths.append(F1_measure_both)


def get_accuracy(TP, FN, FP, TN):
    return (TP + TN) / (TP + FP + TN + FN)


def get_recall(TP, FN, FP, TN):
    return TP / (TP + FN)


def get_precision(TP, FN, FP, TN):
    return TP / (TP + FP)


def get_f1_measure(beta, precision, recall):
    return (1 + beta * beta) * precision * recall / ((beta * beta) * precision + recall)


def get_test_set_label_ne_op(filename):
    """
    输入数据集，得到rating 小于3.1的为 negative 大于 3.1 的为正例
    :param filename:
    :return:
    """
    jsonPath = '../dataset/Amazon/' + filename
    test_set_label_ne_op = []
    o = open(jsonPath, 'r')
    for line_json in open(jsonPath, 'r'):
        line = json.loads(line_json)
        if 'reviewText' in line.keys():  # to avoid json without review
            overall = line['overall']
            if overall < 3.1:
                test_set_label_ne_op.append(0)
            else:
                test_set_label_ne_op.append(1)
    return test_set_label_ne_op


def validation_choose():
    """
    处理已经跑了7个小时的结果，得到合适的top-k，和threshold
    :return:
    """
    result_vali_propagation = np.load(
        '../dataset/Amazon/result/lexicon_semi/k-fold/validation/' + '1_' + 'vali_propagation_result.npy',
        allow_pickle=True).item()
    result_vali_axis = np.load(
        '../dataset/Amazon/result/lexicon_semi/k-fold/validation/' + '1_' + 'vali_axis_result.npy',
        allow_pickle=True).item()
    propagation_averages = []
    axis_averages = []

    pro_temp = 0
    axis_temp = 0
    for i in range(80):
        pro_temp += result_vali_propagation['Acurracy_boths'][i]
        axis_temp += result_vali_axis['Acurracy_eithers'][i]
        if (i + 1) % 20 == 0:
            pro_temp = pro_temp / 20
            axis_temp = axis_temp / 20
            propagation_averages.append(pro_temp)
            axis_averages.append(axis_temp)
            pro_temp = 0
            axis_temp = 0
    print(propagation_averages)  # 第一个最大，则 top-k选择10
    print(axis_averages)  # 第一个最大，则threshold选 0.3
    # [0.6078259302892443, 0.5697734093627551, 0.5146718005973665, 0.4774683342366228]
    # [0.8104256813364943, 0.8101483964142165, 0.7673134600891423, 0.33393642584688227]


if __name__ == '__main__':
    # train_test_split_lexicon(5)
    # get_test_set_label_ne_op('k-fold-dataset/Luxury_Beauty_5' + '_' + 'test' + '_' + str(
    #     1) + '.json')
    # result = np.load('./dataset/Amazon/result/lexicon_semi/k-fold/' + 'result.npy', allow_pickle=True).item()
    # result = np.load(
    #     './dataset/Amazon/result/lexicon_semi/k-fold/validation/' + '1_' + 'vali_propagation_result.npy',
    #     allow_pickle=True).item()
    # evaluation_semi(5)

    # validation_dataset_processed(5)
    # validation_propagation(5)
    # validation_axis(5)
    validation_choose()
