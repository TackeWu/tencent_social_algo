# coding=utf-8
import pandas as pd
import numpy as np
from D_depend import *
from joblib import dump, load
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import time
import os

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
trainY = train.label.values

feature_train = load("./dump/new_train.joblib_dat")
feature_test = load('./dump/new_test.joblib_dat')

# --------------------------去掉第30天------------------------
feature_train = feature_train[train.clickTime.values >=210000]
trainY = trainY[train.clickTime.values >=210000]


print('# ------------------Cross validation..-----------------------')

max_depth = 14
n_splits = 3
test_size = 0.3
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

# -------------------------------Model Fit-----------------------------------------

print('# ReFit model to all data..')
model.fit(feature_train, trainY)


### ------------------------Prediction-------------------------------------

print('# Making predictions on test..')
test_prediction = model.predict_proba(feature_test.values)
test_prediction = test_prediction[:, 1]
col2 = pd.DataFrame(test_prediction, columns=['prob'])
col1 = pd.DataFrame(np.arange(1, test_prediction.shape[0] + 1), columns=['instanceID'])

ans = pd.concat([col1, col2], axis=1)
ans.to_csv('./submit/submission.csv', index=False)
