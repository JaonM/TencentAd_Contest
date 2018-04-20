# -*- coding:utf-8 -*-
"""
feature extract eg. categorical to vector
"""
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np

numeric_features = ['age']

categorical_features = ['gender', 'education', 'consumptionAbility', 'LBS', 'carrier', 'house',
                        'advertiserId', 'campaignId', 'adCategoryId', 'creativeSize', 'productId', 'productType']

multi_categorical_features = ['marriageStatus', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5',
                              'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3', 'appIdInstall', 'appIdAction',
                              'creativeId', 'ct', 'os']


def categorical2vector(columns):
    """
    convert categorical feature to discrete vector
    :param columns: pandas Series or pandas DataFrame
    :return: ndarray shape is (num_samples,classes)
    """
    enc = OneHotEncoder()
    labelEnc = LabelEncoder()
    if isinstance(columns, pd.DataFrame):
        columns.fillna(0, inplace=True)  # handle missing value fill with 0 or -1?
        return enc.fit_transform(labelEnc.fit_transform(columns).toarray())
    elif isinstance(columns, pd.Series):
        columns.fillna(0, inplace=True)
        # return enc.fit_transform(labelEnc.fit_transform(columns.values).reshape(1,-1))
        return labelEnc.fit_transform(columns.values)
    else:
        raise TypeError


def multicategorical2vector(column):
    """
    convert multi-categorical feature to discrete vector
    :param column: pandas Series
    :return: ndarray shape is (num_samples,classes)
    """
    pass


if __name__ == '__main__':
    df_train = pd.read_csv('../input/train_raw.csv', encoding='utf-8', dtype=object)
    print(df_train['LBS'].value_counts())
    # print(categorical2vector(df_train['LBS']))
    # print(np.isfinite(df_train['LBS'].values().any()))
