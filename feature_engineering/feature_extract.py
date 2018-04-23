# -*- coding:utf-8 -*-
"""
feature extract eg. categorical to vector
"""
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import gc
import sys

numeric_features = ['age']

categorical_features = ['gender', 'education', 'consumptionAbility', 'LBS', 'carrier', 'house']

multi_categorical_features = ['marriageStatus', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5',
                              'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3', 'creativeId', 'ct', 'os']

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


if __name__ == '__main__':
    df_train = pd.read_csv('../input/train_clean.csv', encoding='utf-8', dtype=object)
    print(df_train['interest1'].value_counts())
    # print(df_train['LBS'].value_counts())
    # print(multicategorical2vector(df_train[multi_categorical_features], column_names=multi_categorical_features))

    # extract categorical features
    # cats = categorical2vector(df_train[categorical_features], categorical_features)
    # df = pd.DataFrame(data=cats, dtype='int32')
    # df.to_csv('../input/categorical_features.csv', index=False, encoding='utf-8')
