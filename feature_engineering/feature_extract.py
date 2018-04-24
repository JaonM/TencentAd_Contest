# -*- coding:utf-8 -*-
"""
feature extract eg. categorical to vector
"""
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import gc
import sys

numeric_features = ['age']

categorical_features = ['gender', 'education', 'consumptionAbility', 'LBS', 'carrier', 'house']

multi_categorical_features = ['marriageStatus', 'creativeId', 'ct', 'os']

tfidf_features = ['interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1',
                  'topic2', 'topic3']

id_features = ['advertiserId', 'campaignId', 'adCategoryId', 'creativeSize', 'productId', 'productType']

user_action_features = ['appIdInstall', 'appIdAction']


def categorical2vector(columns, column_names=None):
    """
    convert categorical ID feature to discrete vector(one hot)
    :param columns: pandas Series or pandas DataFrame
    :param column_names: column list name
    :return: ndarray shape is (num_samples,classes)
    """
    enc = OneHotEncoder()
    labelEnc = LabelEncoder()
    if isinstance(columns, pd.DataFrame) and column_names is not None:
        columns.fillna(0, inplace=True)  # handle missing value fill with 0 or -1?
        # return enc.fit_transform(labelEnc.fit_transform(columns)).toarray()
        a = np.zeros(shape=(len(columns), 1))
        for column_name in column_names:
            label_array = labelEnc.fit_transform(columns[column_name]).reshape(-1, 1)
            a = np.concatenate((a, label_array), axis=1)
        a = np.delete(a, 0, axis=1)
        return enc.fit_transform(a).toarray()
    elif isinstance(columns, pd.Series):
        columns.fillna(0, inplace=True)
        # return labelEnc.fit_transform(columns.values).reshape(1,-1)
        return enc.fit_transform(labelEnc.fit_transform(columns.values).reshape(-1, 1)).toarray()
    else:
        raise TypeError


def multicategorical2vector(columns, column_names=None):
    """
    convert multi-categorical ID feature to discrete vector
    :param columns: pandas Series
    :param column_names: column list name
    :return: ndarray shape is (num_samples,classes)
    """
    columns.fillna(0, inplace=True)
    mlb = MultiLabelBinarizer()
    if isinstance(columns, pd.DataFrame) and column_names is not None:
        a = None
        for column_name in column_names:
            column = columns[column_name]
            column = column.apply(lambda x: x.split())
            if a is None:
                a = mlb.fit_transform(column.values)
                print('column name is ' + column_name)
                print('a shape is ' + str(a.shape))
                print('size of vector a is ' + str(sys.getsizeof(a) / (1024 * 1024)))
            else:
                tmp = mlb.fit_transform(column.values)
                a = np.concatenate((a, tmp), axis=1)
                print('column name is ' + column_name)
                print('a shape is ' + str(a.shape))
                print('size of vector a is ' + str(sys.getsizeof(a) / (1024 * 1024)))
                del tmp
                gc.collect()
        return a
    elif isinstance(columns, pd.Series):
        print(columns.value_counts())
        columns = columns.apply(lambda x: x.split())
        return mlb.fit_transform(columns.values)
    else:
        raise TypeError


def extract_categorical_features(df_train, categorical_features):
    """
    extract categorical features and store as csv file
    :param df_train: source file
    :param categorical_features: categorical features list
    :return:
    """
    cat_mat = categorical2vector(df_train[categorical_features], categorical_features)
    df_cat = pd.DataFrame(data=cat_mat, dtype='int8')
    df_cat.to_csv('../input/train_categorical_features.csv', index=False, encoding='utf-8')


def extract_id_features(df_train, id_features):
    """
    use label encoder to encode id features
    :param df_train:
    :param id_features:
    :return:
    """
    lb = LabelEncoder()
    a = np.zeros(shape=(len(df_train), 1))
    for _id in id_features:
        column = df_train[_id]
        column = lb.fit_transform(column).reshape(-1, 1)
        a = np.concatenate((a, column), axis=1)
    a = np.delete(a, 0, axis=1)
    df_id = pd.DataFrame(data=a, dtype='int16')
    df_id.to_csv('../input/train_id_features.csv', index=False, encoding='utf-8')


def extract_multicategorical_features(df_train, multicategorical_features):
    """
    use multilabelbinarizer to encode features
    :param df_train:
    :param multicategorical_features:
    :return:
    """
    multi_mat = multicategorical2vector(df_train[multicategorical_features], multicategorical_features)
    df_multi = pd.DataFrame(data=multi_mat, dtype='int8')
    df_multi.to_csv('../input/train_multi_categorical_features.csv', index=False, encoding='utf-8')


def extract_tfidf_features(column):
    """
    extract tfidf features
    :param column:
    :return:
    """
    tfidfVec = TfidfVectorizer(
        ngram_range=(1, 1),
        analyzer='word',
        min_df=1000
    )
    return tfidfVec.fit_transform(column)


if __name__ == '__main__':
    df_train = pd.read_csv('../input/train_clean.csv', encoding='utf-8', dtype=object)
    df_test = pd.read_csv('../input/test_clean.csv',encoding='utf-8',dtype=object)
    # print(df_train['aid'].value_counts())
    # print(df_train['LBS'].value_counts())
    # print(multicategorical2vector(df_train[multi_categorical_features], column_names=multi_categorical_features))

    # extract categorical features
    extract_categorical_features(df_train, categorical_features)

    # extract id features
    # extract_id_features(df_train, id_features)

    # extract multi-categorical features
    # extract_multicategorical_features(df_train, multi_categorical_features)
