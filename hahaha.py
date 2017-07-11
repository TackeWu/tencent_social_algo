import pandas as pd
import numpy as np
import time
from A_3_feature_engine import *
from D_depend import *
from joblib import dump, load
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

raw_train = load('./dump/train.joblib_dat')
raw_test = load('./dump/test.joblib_dat')

# process  cate camgaignID for  train and test !!!

# col_name = ['camgaignID', 'advertiserID', 'creativeID', 'adID']
# clilc_rate = [
#     [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.4],
#     [0, 0.03, 0.11],
#     [0, 0.05, 0.07, 0.09, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30, 0.35, 0.4],
#     [0, 0.05, 0.07, 0.09, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30, 0.35, 0.4]
# ]
# for i in range(len(col_name)):
#     print("--------------->分类" + col_name[i])
#     raw_train, raw_test = cate_(raw_train, raw_test, None, None, col_name[i], clilc_rate[i])

# process train set

print("--------------->统计train is_app_actions")
app_actions = pd.read_csv('./data/user_app_actions.csv')
app_actions['is_app_actions'] = 1
raw_train = pd.merge(raw_train, app_actions, on=['userID', 'appID'], sort=False, how='left')
raw_train['is_app_actions'] = raw_train['is_app_actions'].fillna(0)
raw_train['is_app_actions'] = raw_train['is_app_actions'].astype('int')

raw_train['town_sheng'] = (raw_train.hometown / 100).astype(int)
raw_train['residence_sheng'] = (raw_train.residence / 100).astype(int)
raw_train['con_tele'] = (raw_train.connectionType * 10 + raw_train.telecomsOperator).astype(int)
raw_train['age_edu'] = (raw_train.education * 100 + raw_train.age).astype(int)
raw_train['hour'] = (raw_train.clickTime.values / 100).astype(int) % 100

# same for test set

print("--------------->统计test is_app_actions")
raw_test = pd.merge(raw_test, app_actions, on=['userID', 'appID'], sort=False, how='left')
raw_test['is_app_actions'] = raw_test['is_app_actions'].fillna(0)
raw_test['is_app_actions'] = raw_test['is_app_actions'].astype('int')

raw_test['town_sheng'] = (raw_test.hometown / 100).astype(int)
raw_test['residence_sheng'] = (raw_test.residence / 100).astype(int)
raw_test['con_tele'] = (raw_test.connectionType * 10 + raw_test.telecomsOperator).astype(int)
raw_test['age_edu'] = (raw_test.education * 100 + raw_test.age).astype(int)
raw_test['hour'] = (raw_test.clickTime.values / 100).astype(int) % 100

# col = ['connectionType','telecomsOperator','camgaignID',
#        'appID', 'appPlatform', 'is_app_actions','advertiserID',
#        'age', 'gender','education', 'marriageStatus', 'haveBaby', 'hometown',
#        'residence','positionID', 'sitesetID', 'positionType']


col = ['connectionType','telecomsOperator','camgaignID',
       'appID', 'appPlatform', 'is_app_actions','advertiserID',
       'age', 'gender','education', 'marriageStatus', 'haveBaby','hometown',
       'residence','positionID', 'sitesetID', 'positionType','hour','advertiserID',
       'creativeID', 'adID']

raw_train = raw_train[col]
raw_test = raw_test[col]

# process the new value in test
for i in raw_train.columns.values:
    raw_test[(raw_test['hour'].isin(raw_train['hour'].values) == False).values] = 0

# dump(raw_train, "./dump/new_train.joblib_dat")
# dump(raw_test, "./dump/new_test.joblib_dat")

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
trainY = train.label.values

# feature_train = load("./dump/new_train.joblib_dat")
# feature_test = load('./dump/new_test.joblib_dat')

feature_train = raw_train
feature_test = raw_test

# --------------------------去掉第30天------------------------
feature_train = feature_train[train.clickTime.values < 300000]
trainY = trainY[train.clickTime.values < 300000]

print('# ------------------Cross validation..-----------------------')

max_depth = 14
n_splits = 3
test_size = 0.1
train_size = 0.2
# model = RandomForestClassifier(max_depth=max_depth, n_jobs=-1, random_state=777)
model = RandomForestClassifier(n_estimators=32, max_depth=14, min_samples_split=100, min_samples_leaf=10, random_state=0,
                               criterion='entropy', max_features=8, verbose = 1, n_jobs=-1, bootstrap=False)
trn_scores = []
vld_scores = []
sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=train_size, random_state=777)
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
