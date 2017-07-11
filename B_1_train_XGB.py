# coding=utf-8
import pandas as pd
import numpy as np
from D_depend import *
from joblib import dump, load
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import time
import os
import xgboost as xgb

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
trainY =train.iloc[:,:1].values

feature_train = load("./dump/new_train.joblib_dat")
feature_test = load('./dump/new_test.joblib_dat')

# --------------------------去掉第30天------------------------
feature_train = feature_train[train.clickTime.values >=210000]
trainY = trainY[train.clickTime.values >=210000]

feature_train = feature_train.values
feature_test = feature_test.values

n_trees =  25


param = {'max_depth':16, 'eta':0.3, 'objective':'binary:logistic', 'verbose':0,
         'subsample':1.0, 'min_child_weight':50, 'gamma':0,
         'nthread': 16, 'colsample_bytree':.5, 'base_score':0.16, 'seed': 999}

print('# ------------------Cross validation..-----------------------')

max_depth = 14
n_splits = 3
test_size = 0.3
# model = RandomForestClassifier(max_depth=max_depth, n_jobs=-1, random_state=777)
trn_scores = []
vld_scores = []
sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=777)
for i, (t_ind, v_ind) in enumerate(sss.split(feature_train, trainY)):
    print('# Iter {} / {}'.format(i + 1, n_splits))
    x_trn = feature_train[t_ind]
    y_trn = trainY[t_ind]
    x_vld = feature_train[v_ind]
    y_vld = trainY[v_ind]
    dtrain = xgb.DMatrix(x_trn,label=y_trn)
    dvalid = xgb.DMatrix(x_vld,label=y_vld)


    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    plst = list(param.items()) + [('eval_metric', 'logloss')]

    xgb1 = xgb.train(plst, dtrain, n_trees, watchlist)


    score = log_loss(y_trn, xgb1.predict(dtrain))
    trn_scores.append(score)

    score = log_loss(y_vld, xgb1.predict(dvalid))
    vld_scores.append(score)

print("max_depth: %d   n_splits: %d    test_size: %f" % (max_depth, n_splits, test_size))
print('# TRN logloss: {}'.format(np.mean(trn_scores)))
print('# VLD logloss: {}'.format(np.mean(vld_scores)))

# -------------------------------Model Fit-----------------------------------------

print('# ReFit model to all data..')
dtrain = xgb.DMatrix(feature_train,label=trainY)
dvalid = xgb.DMatrix(feature_test)

watchlist = [(dtrain, 'train')]
plst = list(param.items()) + [('eval_metric', 'logloss')]

xgb1 = xgb.train(plst, dtrain, n_trees, watchlist)

### ------------------------Prediction-------------------------------------

print('# Making predictions on test..')
test_prediction = xgb1.predict(dvalid)

col2 = pd.DataFrame(test_prediction, columns=['prob'])
col1 = pd.DataFrame(np.arange(1, test_prediction.shape[0] + 1), columns=['instanceID'])

ans = pd.concat([col1, col2], axis=1)
ans.to_csv('./submit/submission.csv', index=False)

