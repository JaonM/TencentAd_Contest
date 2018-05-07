# -*- coding:utf-8 -*-
"""
lr model
"""
from sklearn.linear_model import LogisticRegression
from model.base import build_features_train


def train_cv(X, y, k_fold):
    """
    cross validation to train model
    :param X:
    :param y:
    :param k_fold:
    :return:
    """