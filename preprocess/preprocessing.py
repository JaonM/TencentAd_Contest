# -*- coding:utf-8 -*-
"""
preprocess the data
"""
import pandas as pd

# df_train = pd.read_csv('../input/train_raw.csv')
# print(len(df_train))
# missing value count per line
# line_missing = df_train.shape[1] - df_train.count(axis=1)
# missing_index = line_missing[line_missing > 12].index
# print(missing_index)
# drop line where missing value count >12
# df_train.drop(index=missing_index, axis=0, inplace=True)
# print(len(df_train))

# df_train.fillna(0, inplace=True)
# print(df_train.isnull())

# df_train.sort_values(by='aid', axis=0, inplace=True)
# df_train.to_csv('../input/train_clean.csv', index=False, encoding='utf-8')

df_test = pd.read_csv('../input/test.csv', encoding='utf-8')
df_test.fillna(0,inplace=True)
df_test.to_csv('../input/test.csv', index=False, encoding='utf-8')
