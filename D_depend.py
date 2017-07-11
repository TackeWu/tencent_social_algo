import pandas as pd
import numpy as np
import time
from joblib import dump, load
from matplotlib import pyplot as plt


# coding=utf-8
# 拼接datafram  横向  pd.concat([train2,newdata],axis=1)
def searchLeft(data, target):
    left = 0
    right = data.shape[0] - 1
    # find left boundary
    while left <= right:
        mid = (int)((right + left) / 2)
        if data[mid] >= target:
            right = mid - 1
        elif data[mid] < target:
            left = mid + 1
    if left < data.shape[0] and data[left] == target:
        return left
    else:
        return -1


def tongji(feature, title, how):
    label = pd.read_csv('./data/train.csv')['label']
    fea_cnt = feature.value_counts()
    fea1_cnt = feature[label.values == 1].value_counts()
    fea_cnt = pd.DataFrame({'index':fea_cnt.index.values,'total_cnt':fea_cnt.values})
    fea1_cnt = pd.DataFrame({'index':fea1_cnt.index.values,'label_1_cnt':fea1_cnt.values})
    mer = pd.merge(fea_cnt, fea1_cnt,how='left',sort=False,on='index')
    mer = mer.fillna(0)
    plt.figure(title)
    plt.subplot(211)
    plt.plot(mer.index.values, mer.label_1_cnt / mer.total_cnt, how)
    plt.subplot(212)
    plt.bar(mer.index.values, mer.total_cnt)

