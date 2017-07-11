import pandas as pd
import numpy as np
import time
from A_3_feature_engine import *
from D_depend import *
from joblib import dump, load

raw_train = load('./dump/train.joblib_dat')
raw_test = load('./dump/test.joblib_dat')

# process  cate camgaignID for  train and test !!!
#
# col_name = ['camgaignID', 'advertiserID','creativeID','adID']
# clilc_rate = [
#     [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.4],
#     [0,0.03,0.11],
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

col = ['connectionType','telecomsOperator','camgaignID',
'appID', 'appPlatform', 'is_app_actions','advertiserID',
'age', 'gender','education', 'residence', 'sitesetID', 'positionType']

# raw_train = raw_train[col]
# raw_test = raw_test[col]

# process the new value in test
# for i in raw_train.columns.values:
#     raw_test[raw_test[i].isin(raw_train[i].values) == False] = 0

dump(raw_train, "./dump/new_train.joblib_dat")
dump(raw_test, "./dump/new_test.joblib_dat")
