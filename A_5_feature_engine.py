import pandas as pd
import numpy as np
import time
from joblib import dump,load
from A_3_feature_engine import *

print("*****************读取train、test***********************")
start = time.time()

train = load("./dump/train.joblib_dat")
test  = load("./dump/test.joblib_dat")

print("*****************读取完成**************************")
end = time.time()
print("读取用时",end-start)

print("--------------->统计train is_app_actions")
app_actions = pd.read_csv('./data/user_app_actions.csv')
app_actions['is_app_actions'] = 1
train = pd.merge(train,app_actions,on=['userID','appID'],sort=False,how='left')
train['is_app_actions'] = train['is_app_actions'].fillna(0)
train['is_app_actions'] = train['is_app_actions'].astype('int')

print("--------------->统计test is_app_actions")
test = pd.merge(test,app_actions,on=['userID','appID'],sort=False,how='left')
test['is_app_actions'] = test['is_app_actions'].fillna(0)
test['is_app_actions'] = test['is_app_actions'].astype('int')

col_name = ['camgaignID','advertiserID']
clilc_rate =[
    [0, 0.1, 0.4],
    [0,0.03,0.11]
]
for i in range(len(col_name)):
    print("--------------->分类"+col_name[i])
    train,test = cate_(train, test, None,None,col_name[i],clilc_rate[i])



