# -*- coding:utf-8 -*-
"""
base common module
"""
import pandas as pd
import numpy as np
from feature_engineering.feature_extract import extract_positive_probability_single, \
    extract_positive_probability_each_aid_single, extract_avg_probability_each_aid_multi
from feature_engineering.feature_extract import extract_max_probability_each_aid_multi, extract_aid_history_feature, \
    extract_tfidf_features_by_aid, extract_tfidf_features, extract_count_matrix_by_aid
from scipy import sparse
import functools
import gc
import os
from time import time

categorical_features = ['age', 'gender', 'education', 'consumptionAbility', 'LBS', 'carrier', 'house']

multi_categorical_features = ['marriageStatus', 'ct', 'os']

tfidf_features = ['interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1',
                  'topic2', 'topic3']

id_features = ['creativeId', 'advertiserId', 'campaignId', 'adCategoryId', 'creativeSize', 'productId', 'productType']

user_action_features = ['appIdInstall', 'appIdAction']


def _build_features_train(train):
    """
    build train data set features
    :param train:
    :return:
    """
    y = train['label']
    # building statistics features
    print('start building statistics features...')
    # l = ['carrier', 'house', 'creativeId', 'advertiserId', 'campaignId', 'adCategoryId', 'creativeSize', 'productId',
    #      'productType']
    # for feature in l:
    #     X = pd.DataFrame()
    #     df = pd.read_csv('../input/statistics/statics_' + feature + '.csv', encoding='utf-8')
    #     column_index = train.columns.get_loc(feature)
    #     f = functools.partial(extract_positive_probability_single, df_statics=df, column_index=column_index)
    #     X[feature] = train.apply(f, axis=1, raw=True)
    #     X['aid'] = train['aid']
    #     X['uid'] = train['uid']
    #     X.to_csv('../input/merge/' + feature + '.csv', index=False, encoding='utf-8')

    l = ['interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1',
         'topic2', 'topic3', 'appIdInstall', 'appIdAction']
    for feature in l:
        X = pd.DataFrame()
        df = pd.read_csv('../input/statistics/statics_' + feature + '.csv', encoding='utf-8')
        column_index = train.columns.get_loc(feature)
        f = functools.partial(extract_max_probability_each_aid_multi, df_statics=df, column_index=column_index)
        X[feature] = train.apply(f, axis=1, raw=True)
        X['aid'] = train['aid']
        X['uid'] = train['uid']
        X.to_csv('../input/merge/' + feature + '.csv', index=False, encoding='utf-8')
    return X, y


def build_features_train(split_index=None, save=False, load=False):
    """
    building train features
    :param split_index:
    :param save:
    :param load:
    :return:
    """
    if len(os.listdir('../input/all/train')) > 0 and load:
        df_train = pd.read_csv('../input/all/train_clean.csv',
                               encoding='utf-8')
        y = df_train['label']
        X = pd.read_csv('../input/split_' + str(split_index) + '/train/train_vectors.csv', encoding='utf-8')
        X = X.as_matrix()
        for feature in tfidf_features:
            print('doing {} features'.format(feature))
            tfidf_matrix = extract_count_matrix_by_aid(df_train, feature)
            if X is not None and X.shape[0] != tfidf_matrix.shape[0]:
                print('features shape error')
                raise ValueError
            if X is not None:
                X = sparse.hstack((X, tfidf_matrix))
            else:
                X = tfidf_matrix
        # add user app install features
        print('doing app install id count features..')
        tfidf_matrix = extract_count_matrix_by_aid(df_train, 'appIdInstall')
        if X.shape[0] != tfidf_matrix.shape[0]:
            print('features shape error')
            raise ValueError
        X = sparse.hstack((X, tfidf_matrix))
        return X, y

    start_time = time()
    # df_train = pd.read_csv('../input/split_' + str(split_index) + '/train_split_' + str(split_index) + '.csv',
    #                        encoding='utf-8')
    df_train = pd.read_csv('../input/all/train_clean.csv',encoding='utf-8')
    # df_train_corpus = pd.read_csv('../input/all/train_clean.csv', encoding='utf-8')
    y = df_train['label']
    print('start building statistics features')
    X = pd.DataFrame()


    # building history aid statistics feature
    df_positive_corpus = df_train[df_train['label'] == 1]
    print('doing aid history positive probability feature')
    f = functools.partial(extract_aid_history_feature, df_train_vc=df_train['aid'].value_counts(),
                          df_positive_vc=df_positive_corpus['aid'].value_counts())
    X['aid_probability'] = df_train.apply(f, axis=1, raw=True)

    # building categorical features
    for feature in categorical_features:
        print('doing {} feature'.format(feature))
        # df_statistics = pd.read_csv('../input/split_' + str(split_index) + '/statistics/statistics_' + feature + '.csv',
        #                             encoding='utf-8')
        df_statistics = pd.read_csv('../input/all/statistics/statics_' + feature + '.csv', encoding='utf-8')
        df_statistics.set_index('aid', inplace=True)
        column_index = df_train.columns.get_loc(feature)
        f = functools.partial(extract_positive_probability_each_aid_single, df_statistics=df_statistics,
                              column_index=column_index)
        X['aid_' + feature] = df_train.apply(f, axis=1, raw=True)

    # building id statistics features
    for feature in id_features:
        print('doing {} feature'.format(feature))
        # df_statistics = pd.read_csv('../input/split_' + str(split_index) + '/statistics/statistics_' + feature + '.csv',
        #                             encoding='utf-8')
        df_statistics = pd.read_csv('../input/all/statistics/statics_' + feature + '.csv', encoding='utf-8')
        df_statistics.set_index('value', inplace=True)
        column_index = df_train.columns.get_loc(feature)
        f = functools.partial(extract_positive_probability_single, df_statistics=df_statistics,
                              column_index=column_index)
        X[feature] = df_train.apply(f, axis=1, raw=True)

    # building multi-categorical statistics features
    for feature in multi_categorical_features:
        # print('doing {} feature'.format(feature))
        # df_statistics = pd.read_csv('../input/split_' + str(split_index) + '/statistics/statistics_' + feature + '.csv',
        #                             encoding='utf-8')
        df_statistics = pd.read_csv('../input/all/statistics/statics_' + feature + '.csv', encoding='utf-8')
        df_statistics.set_index('aid', inplace=True)
        column_index = df_train.columns.get_loc(feature)
        f = functools.partial(extract_max_probability_each_aid_multi, df_statistics=df_statistics,
                              column_index=column_index)
        f_avg = functools.partial(extract_avg_probability_each_aid_multi, df_statistics=df_statistics,
                                  column_index=column_index)
        X['aid_' + feature] = df_train.apply(f, axis=1, raw=True)
        X['aid_avg_'+feature] = df_train.apply(f_avg,axis=1,raw=True)

    for feature in tfidf_features:
        print('doing {} feature'.format(feature))
        # df_statistics = pd.read_csv('../input/split_' + str(split_index) + '/statistics/statistics_' + feature + '.csv',
        #                             encoding='utf-8')
        df_statistics = pd.read_csv('../input/all/statistics/statics_' + feature + '.csv', encoding='utf-8')
        df_statistics.set_index('aid', inplace=True)
        column_index = df_train.columns.get_loc(feature)
        f = functools.partial(extract_max_probability_each_aid_multi, df_statistics=df_statistics,
                              column_index=column_index)
        X['aid_' + feature] = df_train.apply(f, axis=1, raw=True)

        f_avg = functools.partial(extract_avg_probability_each_aid_multi, df_statistics=df_statistics,
                                  column_index=column_index)
        X['aid_avg_' + feature] = df_train.apply(f_avg, axis=1, raw=True)

    for feature in user_action_features:
        print('doing {} feature'.format(feature))
        # df_statistics = pd.read_csv('../input/split_' + str(split_index) + '/statistics/statistics_' + feature + '.csv',
        #                             encoding='utf-8')
        df_statistics = pd.read_csv('../input/all/statistics/statics_' + feature + '.csv', encoding='utf-8')
        df_statistics.set_index('aid', inplace=True)
        column_index = df_train.columns.get_loc(feature)
        f = functools.partial(extract_max_probability_each_aid_multi, df_statistics=df_statistics,
                              column_index=column_index)
        X['aid_' + feature] = df_train.apply(f, axis=1, raw=True)
        f_avg = functools.partial(extract_avg_probability_each_aid_multi, df_statistics=df_statistics,
                                  column_index=column_index)
        X['aid_avg_' + feature] = df_train.apply(f_avg, axis=1, raw=True)

    # building categorical one-hot features
    # print('doing categorical features one hot')
    # train_categorical_features = pd.read_csv(
    #     '../input/split_' + str(split_index) + '/vectors/train_split_' + str(split_index) + '_categorical_features.csv',
    #     encoding='utf-8')
    # if X.shape[0] != train_categorical_features.shape[0]:
    #     print('features shape error')
    #     raise ValueError
    # X = pd.concat((X, train_categorical_features), axis=1)
    # del train_categorical_features
    # gc.collect()

    # building id label encoder features
    # print('doing id label features')
    # train_id_features = pd.read_csv(
    #     '../input/split_' + str(split_index) + '/vectors/train_split_' + str(split_index) + '_id_features.csv',
    #     encoding='utf-8')
    # if X.shape[0] != train_id_features.shape[0]:
    #     print('features shape error')
    #     raise ValueError
    # X = pd.concat((X, train_id_features), axis=1)
    # del train_id_features
    # gc.collect()

    # building multi-categorical features
    # print('doing multi categorical features')
    # train_multi_categorical_features = pd.read_csv(
    #     '../input/split_' + str(split_index) + '/vectors/train_split_' + str(
    #         split_index) + '_multi_categorical_features.csv', encoding='utf-8')
    # if X.shape[0] != train_multi_categorical_features.shape[0]:
    #     print('features shape error')
    #     raise ValueError
    # X = pd.concat((X, train_multi_categorical_features), axis=1)
    # del train_multi_categorical_features
    # gc.collect()

    # save statistics features to file
    if save:
        X.to_csv('../input/split_' + str(split_index) + '/train/train_vectors.csv', encoding='utf-8', index=False)
    # del X
    # gc.collect()

    # building tfidf features
    # X = X.as_matrix()
    # print('doing count matrix features by each aid')
    # for feature in tfidf_features:
    #     print('doing {} features'.format(feature))
    #     tfidf_matrix = extract_tfidf_features_by_aid(df_train, feature)
    #     if X is not None and X.shape[0] != tfidf_matrix.shape[0]:
    #         print('features shape error')
    #         raise ValueError
    #     if X is not None:
    #         X = sparse.hstack((X, tfidf_matrix))
    #     else:
    #         X = tfidf_matrix
    # # add user app install features
    # print('doing app install id count features..')
    # tfidf_matrix = extract_tfidf_features_by_aid(df_train, 'appIdInstall')
    # if X.shape[0] != tfidf_matrix.shape[0]:
    #     print('features shape error')
    #     raise ValueError
    # X = sparse.hstack((X, tfidf_matrix))
    # if save:
    #     _X = pd.DataFrame(data=X.todense())
    #     _X.to_csv('../input/split_' + str(split_index) + '/train/train_tfidf.csv', encoding='utf-8', index=False)
    print('time used', time() - start_time)
    return X, y


def build_feature_test(df_test):
    """
    build test set features
    :param df_test:
    :return:
    """
    print('start building statistics features')
    X = pd.DataFrame()
    df_train_corpus = pd.read_csv('../input/all/train_clean.csv', encoding='utf-8')
    # building history aid statistics feature
    df_positive_corpus = df_train_corpus[df_train_corpus['label'] == 1]
    print('doing aid history positive probability feature')
    f = functools.partial(extract_aid_history_feature, df_train_vc=df_train_corpus['aid'].value_counts(),
                          df_positive_vc=df_positive_corpus['aid'].value_counts())
    X['aid_probability'] = df_test.apply(f, axis=1, raw=True)

    # building categorical features
    for feature in categorical_features:
        print('doing {} feature'.format(feature))
        df_statistics = pd.read_csv('../input/split_' + str(split_index) + '/statistics/statistics_' + feature + '.csv',
                                    encoding='utf-8')
        df_statistics.set_index('aid', inplace=True)
        column_index = df_train.columns.get_loc(feature)
        f = functools.partial(extract_positive_probability_each_aid_single, df_statistics=df_statistics,
                              column_index=column_index)
        X['aid_' + feature] = df_train.apply(f, axis=1, raw=True)

    # building id statistics features
    for feature in id_features:
        print('doing {} feature'.format(feature))
        df_statistics = pd.read_csv('../input/split_' + str(split_index) + '/statistics/statistics_' + feature + '.csv',
                                    encoding='utf-8')
        df_statistics.set_index('value', inplace=True)
        column_index = df_train.columns.get_loc(feature)
        f = functools.partial(extract_positive_probability_single, df_statistics=df_statistics,
                              column_index=column_index)
        X[feature] = df_train.apply(f, axis=1, raw=True)

    # building multi-categorical statistics features
    for feature in multi_categorical_features:
        print('doing {} feature'.format(feature))
        df_statistics = pd.read_csv('../input/split_' + str(split_index) + '/statistics/statistics_' + feature + '.csv',
                                    encoding='utf-8')
        df_statistics.set_index('aid', inplace=True)
        column_index = df_train.columns.get_loc(feature)
        f = functools.partial(extract_max_probability_each_aid_multi, df_statistics=df_statistics,
                              column_index=column_index)
        X['aid_' + feature] = df_train.apply(f, axis=1, raw=True)

    for feature in tfidf_features:
        print('doing {} feature'.format(feature))
        df_statistics = pd.read_csv('../input/split_' + str(split_index) + '/statistics/statistics_' + feature + '.csv',
                                    encoding='utf-8')
        df_statistics.set_index('aid', inplace=True)
        column_index = df_train.columns.get_loc(feature)
        f = functools.partial(extract_max_probability_each_aid_multi, df_statistics=df_statistics,
                              column_index=column_index)
        X['aid_' + feature] = df_train.apply(f, axis=1, raw=True)

    for feature in user_action_features:
        print('doing {} feature'.format(feature))
        df_statistics = pd.read_csv('../input/split_' + str(split_index) + '/statistics/statistics_' + feature + '.csv',
                                    encoding='utf-8')
        df_statistics.set_index('aid', inplace=True)
        column_index = df_train.columns.get_loc(feature)
        f = functools.partial(extract_max_probability_each_aid_multi, df_statistics=df_statistics,
                              column_index=column_index)
        X['aid_' + feature] = df_train.apply(f, axis=1, raw=True)

    # building categorical one-hot features
    # print('doing categorical features one hot')
    # train_categorical_features = pd.read_csv(
    #     '../input/split_' + str(split_index) + '/vectors/train_split_' + str(split_index) + '_categorical_features.csv',
    #     encoding='utf-8')
    # if X.shape[0] != train_categorical_features.shape[0]:
    #     print('features shape error')
    #     raise ValueError
    # X = pd.concat((X, train_categorical_features), axis=1)
    # del train_categorical_features
    # gc.collect()

    # building id label encoder features
    print('doing id label features')
    train_id_features = pd.read_csv(
        '../input/split_' + str(split_index) + '/vectors/train_split_' + str(split_index) + '_id_features.csv',
        encoding='utf-8')
    if X.shape[0] != train_id_features.shape[0]:
        print('features shape error')
        raise ValueError
    X = pd.concat((X, train_id_features), axis=1)
    del train_id_features
    gc.collect()

    # building multi-categorical features
    print('doing multi categorical features')
    train_multi_categorical_features = pd.read_csv(
        '../input/split_' + str(split_index) + '/vectors/train_split_' + str(
            split_index) + '_multi_categorical_features.csv', encoding='utf-8')
    if X.shape[0] != train_multi_categorical_features.shape[0]:
        print('features shape error')
        raise ValueError
    X = pd.concat((X, train_multi_categorical_features), axis=1)
    del train_multi_categorical_features
    gc.collect()

    # save non tf-idf features to file
    if save:
        X.to_csv('../input/split_' + str(split_index) + '/train/train_vectors.csv', encoding='utf-8', index=False)


if __name__ == '__main__':
    # for i in range(0, 20):
    #     print('{}th'.format(i))
    #     X, y = build_features_train(split_index=i, save=True)
    X,y = build_features_train(save=True)
    # print(X.head())
    # df_train = pd.read_csv('../input/split_0/train_split_0.csv', encoding='utf-8')
    # mat = extract_count_matrix_by_aid(df_train,'kw1')
    # print(mat.toarray())
