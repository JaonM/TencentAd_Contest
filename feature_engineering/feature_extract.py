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
from scipy import sparse
import functools

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
    # df_cat.to_csv('../input/train_categorical_features.csv', index=False, encoding='utf-8')
    return df_cat


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
    # df_id.to_csv('../input/train_id_features.csv', index=False, encoding='utf-8')
    return df_id


def extract_multicategorical_features(df_train, multicategorical_features):
    """
    use multilabelbinarizer to encode features
    :param df_train:
    :param multicategorical_features:
    :return:
    """
    multi_mat = multicategorical2vector(df_train[multicategorical_features], multicategorical_features)
    df_multi = pd.DataFrame(data=multi_mat, dtype='int8')
    # df_multi.to_csv('../input/train_multi_categorical_features.csv', index=False, encoding='utf-8')
    return df_multi


def extract_tfidf_features(column):
    """
    extract tf-idf features global
    :param column:
    :return:
    """
    tfidfVec = TfidfVectorizer(
        ngram_range=(1, 1),
        analyzer='word',
        min_df=1000,
        max_features=100
    )
    return tfidfVec.fit_transform(column)


def extract_tfidf_features_by_aid(df, column):
    """
    extract tf-idf features according to each different aid local tf-idf
    :param df: data frame
    :param column: pandas Series
    :return:
    """
    aids = df['aid'].unique()
    # construct vocabulary
    # vocabulary = list()
    # df[column].apply(lambda x: vocabulary.extend(x.split()))
    # vocabulary = set(vocabulary)
    # print('vocabulary size is {}'.format(len(vocabulary)))
    # print(vocabulary)
    tfidfVec = TfidfVectorizer(
        ngram_range=(1, 1),
        analyzer='word',
        min_df=3000,
        # vocabulary=vocabulary
    )
    tfidf_mat = None
    for aid in aids:
        sub_df = df[df['aid'] == aid]
        print('sub data frame shape is {}'.format(sub_df.shape))
        tmp = tfidfVec.fit_transform(sub_df[column])
        print(tmp.shape)
        if tfidf_mat is None:
            tfidf_mat = tmp
        else:
            if tmp.shape[1] > tfidf_mat.shape[1]:
                tfidf_mat = sparse.hstack(
                    (tfidf_mat, np.zeros(shape=(tfidf_mat.shape[0], tmp.shape[1] - tfidf_mat.shape[1]))))
            elif tmp.shape[1] < tfidf_mat.shape[1]:
                tmp = sparse.hstack(
                    (tmp, np.zeros(shape=(tmp.shape[0], tfidf_mat.shape[1] - tmp.shape[1]))))
            else:
                pass
            tfidf_mat = sparse.vstack((tfidf_mat, tmp))
            print('tfidf matrix shape is {}'.format(tfidf_mat.shape))
        del tmp
        gc.collect()
    return tfidf_mat


def extract_probability_features(df, column_name, df_ad):
    """
    extract the global positive probability for every single feature eg.uid , ad features...
    :param df:
    :param column_name:
    :param store_dict
    :return:
    """
    df_positive = df[df['label'] == '1']
    print(len(df_positive))
    column_value_count = df[column_name].value_counts()
    result = []
    for index, item in df_ad.iterrows():
        print('handing line {}'.format(index))
        # total = len(df[df[column_name] == item[column_name]])
        result_dict = dict()
        for value in df_train[column_name].unique():
            positive_count = len(df_positive[df_positive[column_name] == value])
            total = column_value_count[value]
            positive_rate = round(positive_count / total, 8) if total != 0 else 0
            # print(positive_count)
            result_dict[value] = positive_rate
        result.append(result_dict)
    df_statics = pd.DataFrame(data=result, columns=df[column_name].unique(), dtype='float16')
    df_statics['aid'] = df_ad['aid']
    df_statics.to_csv('../input/statics/statics_' + column_name + '.csv', encoding='utf-8', index=False)


def extract_probability_features_each_aid(df_ad=None, column_name=None, df_train=None, df_positive=None):
    """
    extract each positive probability for each aid of a single-value column eg. user features
    :param df_ad:
    :param column_name:
    :param df_train:
    :param df_positive:
    :return:
    """
    # aid = row[0]
    # positive_df = df_positive[df_positive['aid'] == aid]
    # positive_count = len(positive_df[positive_df[column_name] == row[target_index]])
    # aid_value_count = df_train['aid'].value_counts()[row[0]]
    # print('go go')
    # if aid_value_count == 0:
    #     return 0
    # else:
    #     return positive_count / aid_value_count
    result = []
    print('start calculating {} statics'.format(column_name))
    for index, item in df_ad.iterrows():
        print('calculating {}'.format(index))
        positive_df = df_positive[df_positive['aid'] == str(item['aid'])]
        total = len(df_train[df_train['aid'] == str(item['aid'])])
        result_dict = dict()
        for value in df_train[column_name].unique():
            positive_count = len(positive_df[positive_df[column_name] == value])
            if total == 0:
                result_dict[value] = 0
            else:
                result_dict[value] = round(positive_count / total, 8)
            print('{} positive rate is {}'.format(value, result_dict[value]))
        result.append(result_dict)
    df_statics = pd.DataFrame(data=result, columns=df_train[column_name].unique(), dtype='float16')
    df_statics['aid'] = df_ad['aid']
    df_statics.to_csv('../input/statics/statics_' + column_name + '.csv', index=False, encoding='utf-8')
    return df_statics


def extract_probability_features_each_aid_multi(df_ad, df_train, column_name):
    """
    extract each positive probability for each aid of a multi-value column eg. user features-interest ..topic..
    :param df_ad:
    :param df_train:
    :param column_name:
    :return:
    """
    df_positive = df_train[df_train['label'] == '1']
    result = []
    value_list = []
    for value in df_train[column_name].unique():
        value_list.extend(value.split())
    value_set = set(value_list)
    for index, item in df_ad.iterrows():
        print('calculating index {}'.format(index))
        positive_df = df_positive[df_positive['aid'] == str(item['aid'])]
        total = len(df_train[df_train['aid'] == str(item['aid'])])
        result_dict = dict()
        for index, item in positive_df.iterrows():
            for value in value_set:
                if value in item[column_name].split():
                    result_dict[value] = result_dict.get(value, 0) + 1
        for value in value_set:
            result_dict[value] = round(result_dict.get(value, 0) / total, 8) if total != 0 else 0
            print('{} positive rate is {}'.format(value, result_dict[value]))
        result.append(result_dict)
    df = pd.DataFrame(data=result, columns=list(value_set))
    df['aid'] = df_ad['aid']
    df.to_csv('../input/statics/statics_' + column_name + '.csv', index=False, encoding='utf-8')


def extract_max_probability_each_aid_multi(row, df_statics, column_index):
    """
    extract the max probability in each multi-value column like max-pooling
    :param row:
    :param df_statics:
    :param column_index:
    :return:
    """
    aid = row[0]  # str
    column_values = row[column_index].split()
    df_statics = df_statics[df_statics['aid'] == aid]
    max_value = df_statics[column_values].max()
    return max_value


def extract_positive_probability_single(row, df_statics, column_value):
    """
    extract single-value positive probability in df_statics file
    :param row:
    :param df_statics:
    :param column_value:
    :return:
    """
    aid = row[0]
    df_statics = df_statics[df_statics['aid'] == aid]
    return df_statics[column_value]


if __name__ == '__main__':
    df_train = pd.read_csv('../input/train_clean.csv', encoding='utf-8', dtype=object)
    # df_test = pd.read_csv('../input/test_clean.csv', encoding='utf-8', dtype=object)
    # print(df_train['aid'].value_counts())
    # print(df_train['LBS'].value_counts())
    # print(multicategorical2vector(df_train[multi_categorical_features], column_names=multi_categorical_features))

    # extract categorical features
    # df = extract_categorical_features(df_test, categorical_features)
    # df.to_csv('../input/test_categorical_features.csv', encoding='utf-8', index=False)

    # extract id features
    # df = extract_id_features(df_test, id_features)
    # df.to_csv('../input/test_id_features.csv', encoding='utf-8', index=False)

    # extract multi-categorical features
    # df = extract_multicategorical_features(df_test, multi_categorical_features)
    # df.to_csv('../input/test_multi_categorical_features.csv', encoding='utf-8', index=False)
    # mat = extract_tfidf_features_by_aid(df_train, 'interest1')

    # build statics features
    df_ad = pd.read_csv('../input/adFeature.csv', encoding='utf-8')
    df_positive = df_train[df_train['label'] == '1']

    # single value group by aid
    # for feature in ['gender', 'education', 'consumptionAbility', 'LBS', 'carrier', 'house', 'age']:
    #     try:
    #         extract_probability_features_each_aid(df_ad=df_ad, column_name=feature, df_train=df_train,
    #                                               df_positive=df_positive)
    #     except Exception as e:
    #         continue

    # single value
    # for feature in in['advertiserId', 'campaignId', 'adCategoryId', 'creativeSize', 'productId', 'productType']:
    #     try:
    #         extract_probability_features(df_train, feature, df_ad)
    #     except Exception as e:
    #         continue

    # for feature in ['interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1',
    #                 'topic2', 'topic3', 'appIdInstall', 'appIdAction']:
    for feature in ['marriageStatus', 'creativeId', 'ct', 'os', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3',
                    'appIdInstall', 'appIdAction', 'kw1']:
        try:
            extract_probability_features_each_aid_multi(df_ad=df_ad, df_train=df_train, column_name=feature)
        except Exception as e:
            print(e)
            # continue
