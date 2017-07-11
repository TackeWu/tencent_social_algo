import pandas as pd
import numpy as np
import time
from joblib import dump, load

print("*****************读取文件***********************")
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# --------------------查询creativceID信息--------------------------------
print("query the creativeID information ")
start = time.time()

total = pd.concat([train.creativeID, test.creativeID], axis=0)
total = pd.DataFrame(total)
name = total.creativeID.unique()
y = np.ndarray(name.shape,float)
invalid = np.ndarray(name.shape,int)
for j, na in enumerate(name):
    desc = train[train.creativeID.values == na].label.describe()
    y[j] = desc[1]
    if desc[0] < 30:
        invalid[j] = 1
y[np.isnan(y)] = 0
invalid[np.isnan(invalid)] = 1

le_0 = name[np.logical_and(y >= 0.8, y <= 1)]
le_1 = name[np.logical_and(y >= 0.6, y < 0.8)]
le_2 = name[np.logical_and(y >= 0.4, y < 0.6)]
le_3 = name[np.logical_and(y >= 0.2, y < 0.4)]
le_4 = name[np.logical_and(y >= 0.0, y < 0.2)]
le_5 = name[invalid == 1]
# 构造特征基础结构
data_train = np.ndarray((train.shape[0], 1))
data_test = np.ndarray((test.shape[0], 1))
creativeID_info_train = pd.DataFrame(data_train, columns=['creative'])
creativeID_info_test = pd.DataFrame(data_test, columns=['creative'])
# train 分类
creativeID_info_train.loc[train.creativeID.isin(le_0), 'creative'] = 5
creativeID_info_train.loc[train.creativeID.isin(le_1), 'creative'] = 4
creativeID_info_train.loc[train.creativeID.isin(le_3), 'creative'] = 3
creativeID_info_train.loc[train.creativeID.isin(le_4), 'creative'] = 2
creati veID_info_train.loc[train.creativeID.isin(le_5), 'creative'] = 1
creativeID_info_train.loc[train.creativeID.isin(le_5), 'creative'] = 0
# test 分类
creativeID_info_test.loc[test.creativeID.isin(le_0), 'creative'] = 5
creativeID_info_test.loc[test.creativeID.isin(le_1), 'creative'] = 4
creativeID_info_test.loc[test.creativeID.isin(le_3), 'creative'] = 3
creativeID_info_test.loc[test.creativeID.isin(le_4), 'creative'] = 2
creativeID_info_test.loc[test.creativeID.isin(le_5), 'creative'] = 1
creativeID_info_test.loc[test.creativeID.isin(le_5), 'creative'] = 0

creativeID_info_train = creativeID_info_train.astype(int)
creativeID_info_test = creativeID_info_test.astype(int)

end = time.time()
print("creativeID finished in :", end - start)

dump(creativeID_info_train, "./dump/feature_4_train.joblib_dat")
dump(creativeID_info_test, "./dump/feature_4_test.joblib_dat")
