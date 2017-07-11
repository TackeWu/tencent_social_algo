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

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

feature1_train = load("./dump/feature_1_train.joblib_dat")
feature2_train = load("./dump/feature_2_train.joblib_dat")
feature1_test = load('./dump/feature_1_test.joblib_dat')
feature2_test = load('./dump/feature_2_test.joblib_dat')
feature_train = pd.concat([feature1_train, feature2_train], axis=1)
feature_test = pd.concat([feature1_test, feature2_test], axis=1)

print('# ------------------ line validation -----------------------')

Time = [270000,280000]
max_depth = 17
n_splits = 3
test_size = 0.3
model = RandomForestClassifier(max_depth=max_depth, n_jobs=-1, random_state=777)

for time in Time:
    print("--------------> 划分 %d 天" % time)
    feature_cut_train_x = feature_train.ix[np.logical_and(train.clickTime<time,True)]
    feature_cut_train_Y = train.label.ix[np.logical_and(train.clickTime<time,True)]
    feature_cut_test_x  = feature_train.ix[np.logical_and(train.clickTime>=300000,True)]
    feature_cut_test_Y  = train.label.ix[np.logical_and(train.clickTime>=300000,True)]
    model.fit(feature_cut_train_x,feature_cut_train_Y)
    score1  = log_loss(feature_cut_train_Y,model.predict_proba(feature_cut_train_x))
    trn_scores.append(score1)
    score2 = log_loss(feature_cut_test_Y,model.predict_proba(feature_cut_test_x))
    vld_scores.append(score2)
    print("+++++TRN logloss:",score1)
    print("+++++VLD logloss:",score2)

print ("max_depth: %d   n_splits: %d    test_size: %f" % (max_depth, n_splits, test_size))
print('# Mean TRN logloss: {}'.format(np.mean(trn_scores)))
print('# Mean VLD logloss: {}'.format(np.mean(vld_scores)))
