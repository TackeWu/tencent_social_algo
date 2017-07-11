# coding=utf-8
import pandas as pd
import numpy as np
import time
from joblib import dump, load

print("读取数据")
start = time.time()
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
position = pd.read_csv('./data/position.csv')
ad = pd.read_csv('./data/ad.csv')
user = pd.read_csv('./data/user.csv')
end = time.time()
print("读取数据结束, 用时", end - start)

# --------------------查询用户信息--------------------------------
print("query the user's information ")
start = time.time()

user_info_train = pd.merge(train, user, on='userID', sort=False, how='left')
user_info_test = pd.merge(test, user, on='userID', sort=False, how='left')

end = time.time()
print("user's information finished in :", end - start)

# ------------------------查询上下文信息------------------------------------------------------
print("make position")
start = time.time()

position_info_train = pd.merge(train, position, on='positionID', sort=False, how='left')
position_info_test = pd.merge(test, position, on='positionID', sort=False, how='left')

end = time.time()
print("make position finished in , 用时:", end - start)

# ---------------------判断点击时间的小时分成24小时---------------------------------------
print("get the hour user click")
start = time.time()

time_info_train = (train.clickTime.values / 100).astype(int) % 100
time_info_train = pd.DataFrame(time_info_train, columns=['hour'])

time_info_test = (test.clickTime.values / 100).astype(int) % 100
time_info_test = pd.DataFrame(time_info_test, columns=['hour'])

end = time.time()
print("get the hour user click finished in: ", end - start)

user_info_train = user_info_train.drop(['positionID'], axis=1)
user_info_test = user_info_test.drop(['positionID'], axis=1)
feature_train = pd.concat([user_info_train, position_info_train, time_info_train], axis=1)
feature_train = feature_train.ix[:, ['age', 'gender', 'education', 'marriageStatus', 'haveBaby', 'hometown',
                                     'residence', 'positionID', 'sitesetID', 'positionType', 'hour']]
dump(feature_train, "./dump/feature_2_train.joblib_dat")

feature_test = pd.concat([user_info_test, position_info_test, time_info_test], axis=1)

feature_test = feature_test.ix[:, ['age', 'gender', 'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence',
                                   'positionID', 'sitesetID', 'positionType', 'hour']]
dump(feature_test, "./dump/feature_2_test.joblib_dat")
