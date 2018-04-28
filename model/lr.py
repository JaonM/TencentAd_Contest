# -*- coding:utf-8 -*-
"""
lr model
"""

import pandas as pd
import numpy as np
from feature_engineering.feature_extract import id_features, categorical_features, multi_categorical_features, \
    user_action_features
from feature_engineering.feature_extract import extract_positive_probability_single
from feature_engineering.feature_extract import extract_max_probability_each_aid_multi
from scipy import sparse
import functools

df_train_corpus = pd.read_csv('../input/train_clean.csv', encoding='utf-8')
# df_test_corpus = pd.read_csv('../input/test_clean.csv', encoding='utf-8')

# df_test = pd.read_csv('../input/test_small.csv', encoding='utf-8', dtype='int16')
df_train = pd.read_csv('../input/train_small.csv', encoding='utf-8', dtype='int16')


def build_features_train(train):
    """
    build train data set features
    :param train:
    :return:
    """
    y = train['label']
    # building statics features
    print('start building statics features...')
    # for feature in id_features:
    #     df = pd.read_csv('../input/statics/statics_' + feature + '.csv', encoding='utf-8')
    #     column_index = train.columns.get_loc(feature)
    #     f = functools.partial(extract_positive_probability_single, df_statics=df, column_index=column_index)
    #     X[feature] = train.apply(f, axis=1, raw=True)

    # for feature in multi_categorical_features:
    for feature in ['ct', 'os']:
        X = pd.DataFrame()
        df = pd.read_csv('../input/statics/statics_' + feature + '.csv', encoding='utf-8')
        column_index = train.columns.get_loc(feature)
        f = functools.partial(extract_max_probability_each_aid_multi, df_statics=df, column_index=column_index)
        X[feature] = train.apply(f, axis=1, raw=True)
        X['aid'] = train['aid']
        X['uid'] = train['uid']
        X.to_csv('../input/merge/' + feature + '.csv', index=False, encoding='utf-8')
    return X, y


if __name__ == '__main__':
    X, y = build_features_train(df_train_corpus)
    print(X.head())
