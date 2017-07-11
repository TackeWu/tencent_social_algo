import pandas as pd
import numpy as np
import time
from joblib import dump,load


def cate_(train, test, table,con_name,col_name,clilc_rate):   #按点击率分类
    print("---------------------处理",col_name)
    if table == None:
        feature_train = train
        feature_test = test
    else :
        feature_train = pd.merge(train, table, on=con_name, sort=False, how='left')  # 连接train,table
        feature_test = pd.merge(test, table, on=con_name, sort=False, how='left')  # 连接test,table
    feature_True = feature_train[feature_train.label == 1]  # 取所有label为1的train

    feature = feature_train[col_name].value_counts() #计算每个id出现的数量
    value_con = feature_True[col_name].value_counts()  #计算标签为1的每个id的数量
    feature = pd.DataFrame(
        {col_name: feature.index.to_series().values, 'values1': feature.values})  # 把id和每个id数量存入DataFrame
    value_con = pd.DataFrame(
        {col_name: value_con.index.to_series().values, 'values2': value_con.values})  # 把标签为1的每个id和id的数量存入DataFrame
    feature = pd.merge(feature, value_con, on=col_name, how='left', sort=False)
    feature = feature.fillna(0)
    feature['values'] = feature['values2'] / feature['values1']
    feature = feature.ix[:, [col_name, 'values']]
    feature['cate_'+col_name] = 0
    for j,rate in enumerate(clilc_rate):
        feature.ix[feature['values'] > rate, ['cate_'+col_name]] = int(j+1)
    feature = feature.drop('values', axis=1)

    print('**'+col_name+'训练集**')
    feature_train = pd.merge(feature_train, feature, on=col_name, sort=False, how='left')  # 训练集cate
    feature_train = feature_train.fillna(0)

    print('**' + col_name + '测试集**')
    feature_test = pd.merge(feature_test, feature, on=col_name, sort=False, how='left')  # 测试集cate
    feature_test = feature_test.fillna(0)

    return feature_train,feature_test


