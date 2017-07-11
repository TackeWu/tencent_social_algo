import pandas as pd
import numpy as np
import time
from joblib import dump, load
from matplotlib import pyplot as plt

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
trainY = train.label.values
feature1_train = load("./dump/feature_1_train.joblib_dat")
feature2_train = load("./dump/feature_2_train.joblib_dat")
feature1_test = load('./dump/feature_1_test.joblib_dat')
feature2_test = load('./dump/feature_2_test.joblib_dat')
feature_train = pd.concat([feature1_train, feature2_train], axis=1)
feature_test = pd.concat([feature1_test, feature2_test], axis=1)
del feature2_test, feature1_test, feature2_train, feature1_train



# --------------------按照 creativeID 统计-------------------------

name = train.creativeID.unique()
X = np.arange(0, len(name), 1)
y = np.ndarray(X.shape)
for j, na in enumerate(name):
    y[j] = train[train.creativeID.values == na].label.mean()
plt.figure("creativeID统计")
plt.plot(X, y, '.')
# --------------------按照 小时 统计-------------------------

X = np.arange(0, 24, 1)
y = np.ndarray(X.shape)
for j, hour in enumerate(X):
    logi = np.logical_and((train.clickTime.values / 100 % 100).astype(int) >= hour,
                          (train.clickTime.values / 100 % 100).astype(int) <= hour + 2)
    y[j] = train[logi].label.values.mean()
plt.figure("hour统计")
plt.plot(X, y, '-')

# --------------------按照 每天 统计-------------------------
X = np.arange(17, 31)
y = np.ndarray(X.shape)
for i, day in enumerate(X):
    start = day * 10000
    end = (day + 1) * 10000
    logi = np.logical_and(train.clickTime.values >= start, train.clickTime.values <= end)
    y[i] = train[logi].label.values.sum()
plt.figure("day统计")
plt.plot(X, y, '-')

# --------------------按照 分钟 统计-------------------------
X = np.arange(0, 60, 10)
y = np.ndarray(X.shape)
for j, minutes in enumerate(X):
    logi = np.logical_and((train.clickTime.values % 10000).astype(int) >= minutes,
                          (train.clickTime.values % 10000).astype(int) <= minutes + 10)
    y[j] = train[logi].label.values.mean()
plt.figure("分钟统计")
plt.plot(X, y, '*')

# -----------------------年龄统计-----------------------------
X = np.arange(0, 81, 1)
y = np.ndarray(X.shape)
for j, age in enumerate(X):
    y[j] = train[feature_train.age.values == age].label.mean()
plt.figure("年龄统计")
plt.plot(X, y, '-')

# ---------------------教育统计-----------------------
X = np.arange(0, 8, 1)
y = np.ndarray(X.shape)
for j, edu in enumerate(X):
    y[j] = train[feature_train.education.values == edu].label.mean()
plt.figure("教育统计")
plt.plot(X, y, '-')

# ---------------------APPID统计-----------------------
name = feature_train.appID.unique()
X = np.arange(0, 50, 1)
y = np.ndarray(X.shape)
for j, na in enumerate(name):
    y[j] = train[feature_train.appID.values == na].label.mean()
plt.figure("appid统计")
plt.plot(X, y, '*')
plt.xticks(range(len(name)), name)


# --------------------判断周末----------------------
X = np.arange(17, 30)
y = np.ndarray(X.shape)
for i, day in enumerate(X):
    start1 = day * 10000
    end1 = (day + 1) * 10000
    start2 = (day + 1) * 10000
    end2 = (day + 2) * 10000
    y[i] = 0
    logi1 = np.logical_and(train.clickTime.values >= start1, train.clickTime.values <= end1)
    # logi2 = np.logical_or((train.clickTime.values / 100 % 100).astype(int) >= 22, (train.clickTime.values / 100 % 100).astype(int) <= 3)
    logi3 = np.logical_and(logi1, (train.clickTime.values / 100 % 100).astype(int) >= 22)
    y[i] =y[i] + train[logi3].label.value_counts()[0]
    logi1 = np.logical_and(train.clickTime.values >= start2, train.clickTime.values <= end2)
    logi3 = np.logical_and(logi1, (train.clickTime.values / 100 % 100).astype(int) <= 6)
    y[i] =y[i] + train[logi3].label.value_counts()[0]
plt.figure("day22DINA统计")
plt.plot(X, y, '-')
plt.grid(True, linestyle = "-.")
