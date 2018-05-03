# -*- coding:utf-8 -*-
"""
use ensemble under-sampling to process data
"""

import pandas as pd
import os

df_train = pd.read_csv('../input/all/train_clean.csv', encoding='utf-8')

# df_train_negative = df_train[df_train['label'] == -1]
# df_train_positive = df_train[df_train['label'] == 1]
#
# fraction = len(df_train_positive) / len(df_train_negative)
#
# print(int(len(df_train) / (len(df_train_positive))))
#
# for i in range(int(len(df_train) / (len(df_train_positive)))):
#     df_train_negative_sample = df_train_negative.sample(frac=fraction, replace=True)
#     df = pd.concat((df_train_positive, df_train_negative_sample), axis=0)
#     print(len(df))
#     if not os.path.exists('../input/split_' + str(i)):
#         os.mkdir('../input/split_' + str(i))
#
#     # shuffle
#     df = df.sample(frac=1).reset_index(drop=True)
#     print('p n rate is {}'.format(len(df[df['label'] == 1]) / len(df[df['label'] == -1])))
#     df.to_csv('../input/split_' + str(i) + '/train_split_' + str(i) + '.csv', encoding='utf-8', index=False)

# df = pd.read_csv('../input/split_0/train_split_0.csv', encoding='utf-8')
# for i in range(1, 20):
#     _df = pd.read_csv('../input/split_' + str(i) + '/train_split_' + str(i) + '.csv', encoding='utf-8')
#     df = pd.concat((df, _df))
# df = df[df['label'] == -1]
# print(len(df))
# print('after remove duplicate')
# df.drop_duplicates(inplace=True)
# print(len(df))

print(df_train['aid'])