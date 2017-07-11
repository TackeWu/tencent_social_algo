import pandas as pd
import numpy as np
from joblib import dump, load
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import time
import os


def check_log_loss(max_depth, n_splits, test_size):
    model = RandomForestClassifier(max_depth=max_depth, n_jobs=-1, random_state=777)
    trn_scores = []
    vld_scores = []
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=777)
    for i, (t_ind, v_ind) in enumerate(sss.split(feature_train, trainY)):
        print('# Iter {} / {}'.format(i + 1, n_splits))
        x_trn = feature_train.values[t_ind]
        y_trn = trainY[t_ind]
        x_vld = feature_train.values[v_ind]
        y_vld = trainY[v_ind]

        model.fit(x_trn, y_trn)

        score = log_loss(y_trn, model.predict_proba(x_trn))
        trn_scores.append(score)

        score = log_loss(y_vld, model.predict_proba(x_vld))
        vld_scores.append(score)

    print("max_depth: %d   n_splits: %d    test_size: %f" % (max_depth, n_splits, test_size))
    print('# TRN logloss: {}'.format(np.mean(trn_scores)))
    print('# VLD logloss: {}'.format(np.mean(vld_scores)))


train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
trainY = train.label.values
feature1_train = load("./dump/feature_1_train.joblib_dat")
feature2_train = load("./dump/feature_2_train.joblib_dat")
feature1_test = load('./dump/feature_1_test.joblib_dat')
feature2_test = load('./dump/feature_2_test.joblib_dat')
feature_train = pd.concat([feature1_train, feature2_train], axis=1)
feature_test = pd.concat([feature1_test, feature2_test], axis=1)

print('# ------------------start try..-----------------------')

depth = [14]
split = [3, 5, 7, 9, 11, 13]
testsize = [0.5]
for i in depth:
    for j in split:
        for k in testsize:
            check_log_loss(i, j, k)
