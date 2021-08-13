# coding=utf-8
# import os

import numpy as np

import lexicon_based.sentiment_recognition
import log_odds_ratio
import pottscores
import lexicon_based.evaluation


def evaluation_supervised(k):
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
        log_odds_ratio.log_odds_ratio(
            'k-fold-dataset/Luxury_Beauty_5' + '_' + 'train' + '_' + str(flag) + '.json',
            'k-fold/lexicon_dic_log' + '_' + str(flag) + '.npy',
            'k-fold/log_intermediate' + '_' + str(flag) + '.npy', 'log_intermediate_gbc.npy')
        print(str(flag) + 'begin2')
        pottscores.pott_score(
            'k-fold-dataset/Luxury_Beauty_5' + '_' + 'train' + '_' + str(flag) + '.json',
            'k-fold/lexicon_dic_pottscore' + '_' + str(flag) + '.npy')
        print(str(flag) + 'begin3')
        lexicon_based.sentiment_recognition.sentiment_recognition(method='supervised_both', threshold=0,
                                                                  pott_score_dic_filename='k-fold/lexicon_dic_pottscore' + '_' + str(
                                                                      flag) + '.npy',
                                                                  log_odds_ratio_dic_filename='k-fold/lexicon_dic_log' + '_' + str(
                                                                      flag) + '.npy',
                                                                  filename='k-fold-dataset/Luxury_Beauty_5' + '_' + 'test' + '_' + str(
                                                                      flag) + '.json',
                                                                  sentiment_recognition_either_filename='k-fold/sentiment_recognition_log' + '_' + str(
                                                                      flag) + '.npy',
                                                                  sentiment_recognition_both_filename='k-fold/sentiment_recognition_pott' + '_' + str(
                                                                      flag) + '.npy')
        print(str(flag) + 'begin4')
        # 计算accuracy
        sentiment_recognition_either = np.load(
            '../dataset/Amazon/result/lexicon_supervised/k-fold/sentiment_recognition_log' + '_' + str(
                flag) + '.npy')
        sentiment_recognition_both = np.load(
            '../dataset/Amazon/result/lexicon_supervised/k-fold/sentiment_recognition_pott' + '_' + str(
                flag) + '.npy')
        test_set_label_ne_op = lexicon_based.evaluation.get_test_set_label_ne_op(
            'k-fold-dataset/Luxury_Beauty_5' + '_' + 'test' + '_' + str(
                flag) + '.json')
        lexicon_based.evaluation.eva_calculate_general(sentiment_recognition_either, test_set_label_ne_op, TP_eithers,
                                                       FN_eithers, FP_eithers,
                                                       TN_eithers, Acurracy_eithers, Recall_eithers, Precision_eithers,
                                                       F1_measure_eithers)
        lexicon_based.evaluation.eva_calculate_general(sentiment_recognition_both, test_set_label_ne_op, TP_boths,
                                                       FN_boths, FP_boths, TN_boths,
                                                       Acurracy_boths, Recall_boths, Precision_boths, F1_measure_boths)
    result = {'TP_eithers': TP_eithers, 'TP_boths': TP_boths, 'FN_eithers': FN_eithers, 'FN_boths': FN_boths,
              'FP_eithers': FP_eithers, 'FP_boths': FP_boths, 'TN_eithers': TN_eithers, 'TN_boths': TN_boths,
              'Acurracy_eithers': Acurracy_eithers, 'Acurracy_boths': Acurracy_boths, 'Recall_eithers': Recall_eithers,
              'Recall_boths': Recall_boths,
              'Precision_eithers': Precision_eithers, 'Precision_boths': Precision_boths,
              'F1_measure_eithers': F1_measure_eithers, 'F1_measure_boths': F1_measure_boths}
    # 保存intermediate_vb
    np.save('../dataset/Amazon/result/lexicon_supervised/k-fold/' + 'result.npy', result)


def evaluation_supervised_log(k):
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
        print(str(flag) + 'log_begin')
        log_odds_ratio.log_odds_ratio(
            'k-fold-dataset/Luxury_Beauty_5' + '_' + 'train' + '_' + str(flag) + '.json',
            'k-fold/lexicon_dic_log' + '_' + str(flag) + '.npy',
            'k-fold/log_intermediate' + '_' + str(flag) + '.npy', 'log_intermediate_gbc.npy')
        print(str(flag) + 'begin2')
        lexicon_based.sentiment_recognition.sentiment_recognition(method='log_odds_ratio', threshold=0.3225,
                                                                  log_odds_ratio_dic_filename='k-fold/lexicon_dic_log' + '_' + str(
                                                                      flag) + '.npy',
                                                                  filename='k-fold-dataset/Luxury_Beauty_5' + '_' + 'test' + '_' + str(
                                                                      flag) + '.json',
                                                                  sentiment_recognition_either_filename='k-fold/sentiment_recognition_log' + '_' + str(
                                                                      flag) + '.npy')
        print(str(flag) + 'begin4')
        # 计算accuracy
        sentiment_recognition_either = np.load(
            '../dataset/Amazon/result/lexicon_supervised/k-fold/sentiment_recognition_log' + '_' + str(
                flag) + '.npy')
        test_set_label_ne_op = lexicon_based.evaluation.get_test_set_label_ne_op(
            'k-fold-dataset/Luxury_Beauty_5' + '_' + 'test' + '_' + str(
                flag) + '.json')
        lexicon_based.evaluation.eva_calculate_general(sentiment_recognition_either, test_set_label_ne_op, TP_eithers,
                                                       FN_eithers, FP_eithers,
                                                       TN_eithers, Acurracy_eithers, Recall_eithers, Precision_eithers,
                                                       F1_measure_eithers)
    result = {'TP_eithers': TP_eithers, 'TP_boths': TP_boths, 'FN_eithers': FN_eithers, 'FN_boths': FN_boths,
              'FP_eithers': FP_eithers, 'FP_boths': FP_boths, 'TN_eithers': TN_eithers, 'TN_boths': TN_boths,
              'Acurracy_eithers': Acurracy_eithers, 'Acurracy_boths': Acurracy_boths, 'Recall_eithers': Recall_eithers,
              'Recall_boths': Recall_boths,
              'Precision_eithers': Precision_eithers, 'Precision_boths': Precision_boths,
              'F1_measure_eithers': F1_measure_eithers, 'F1_measure_boths': F1_measure_boths}
    # 保存intermediate_vb
    np.save('../dataset/Amazon/result/lexicon_supervised/k-fold/' + 'result_log.npy', result)


if __name__ == '__main__':
    # evaluation_supervised_log(5)
    result = np.load('./dataset/Amazon/result/lexicon_supervised/k-fold/' + 'result_log.npy', allow_pickle=True).item()
