import pandas as pd
import numpy as np
from joblib import dump, load
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import time
import os
import scipy as sp
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll



train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
trainY = train.label.values
feature1_train = load("./dump/feature_1.joblib_dat")
feature2_train = load("./dump/feature_2.joblib_dat")
feature1_test = load('./dump/feature_1_test.joblib_dat')
feature2_test = load('./dump/feature_2_test.joblib_dat')
feature1_train = feature1_train.drop(['creativeID', 'userID'], axis=1)
feature_train = pd.concat([feature1_train, feature2_train], axis=1)
feature_test = pd.concat([feature1_test, feature2_test], axis=1)

print('# ------------------start try..-----------------------')

param_grid = {
    'n_estimators':[20,50,80,100],
    'max_features':[4,5,6,7,8,'float','log2','sqrt'],
    'max_depth':[13,14,15,16,17]
}
model = RandomForestClassifier(n_jobs=-1, random_state=777)
grid_search_rf = GridSearchCV(model,param_grid=param_grid,cv=3,scoring='neg_log_loss')
print("parameters:")
grid_search_rf.fit(feature_train,trainY)
print("Best score: %0.3f" % grid_search_rf.best_score_)
print("Best parameters set:")
best_parameters = grid_search_rf.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

