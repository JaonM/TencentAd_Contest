# -*- coding:utf-8 -*-
"""
use ensemble undersampling to process data
"""

import pandas as pd

df_train = pd.read_csv('../input/train_clean.csv', encoding='utf-8')

# print(len(df_train[df_train['label'] == 1]))

# print(len(df_train[df_train['label'] == -1]))

# fraction = len(df_train[df_train['label'] == 1]) / len(df_train[df_train['label'] == -1])

# print(df_train['creativeId'].value_counts())

for index in df_train['ct'].value_counts().index:
    print(str(index)+' '+str(type(index)))