import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#---------------------user train table----------------------
train = pd.read_csv('./data/train.csv')
user = pd.read_csv('./data/user.csv')
feature= pd.merge(user,train,on='userID')
feature1 = pd.merge(user,train,on='userID')
feature = feature.ix[np.logical_and(feature.label.values == 1,True)]


#----------------------haveBaby plot--------------------------
str1 = feature.haveBaby.astype('int').astype('str')
str2 = feature.label.astype('int').astype('str')
str = str1+ str2
str = str.value_counts()
value_con_index = value_con.index
value_con_values = value_con.values/feature1.haveBaby.shape
plt.figure('haveBaby')
plt.plot(range(len(value_con_index)),value_con_values)
plt.xticks(range(len(value_con_index)),value_con_index)



#----------------------marriageStatus plot---------------------
str3 = feature.marriageStatus.astype('int').astype('str')
str4 = feature.label.astype('int').astype('str')
str5 = str3+ str4
value_con = str5.value_counts()
value_con_index = value_con.index
value_con_values = value_con.values/feature1.marriageStatus.shape
plt.figure('marrign')
plt.plot(range(len(value_con_index)),value_con_values)
plt.xticks(range(len(value_con_index)),value_con_index)




#---------------------ad train table--------------------------

train = pd.read_csv('./data/train.csv')
ad = pd.read_csv('./data/ad.csv')
feature2= pd.merge(ad,train,on='creativeID',sort=False,how='right')
feature1 = feature2[feature2.label == 1]

#----------------------advertiserID plot---------------------
feature = feature2.advertiserID.value_counts()
value_con = feature1.advertiserID.value_counts()
feature = pd.DataFrame({'index':feature.index.to_series().values,'values1':feature.values})
value_con = pd.DataFrame({'index':value_con.index.to_series().values,'values2':value_con.values})
feature = pd.merge(feature,value_con,on='index',how='left')
feature = feature.fillna(0)
feature['values'] = feature['values2']/feature['values1']
plt.figure('advertiserID')
plt.plot(feature.index,feature['values2']/feature['values1'],'.')

#----------------------adID plot------------------------------

feature = feature2.adID.value_counts()
value_con = feature1.adID.value_counts()
feature = pd.DataFrame({'index':feature.index.to_series().values,'values1':feature.values})
value_con = pd.DataFrame({'index':value_con.index.to_series().values,'values2':value_con.values})
feature  = pd.merge(feature,value_con,on='index',how='left')
feature = feature.fillna(0)
plt.figure('adID')
plt.plot(feature.index,feature['values2']/feature['values1'],'.')

#----------------------creativeID plot -----------------------

feature = feature2.creativeID.value_counts()
value_con = feature1.creativeID.value_counts()
feature = pd.DataFrame({'index':feature.index.to_series().values,'values1':feature.values})
value_con = pd.DataFrame({'index':value_con.index.to_series().values,'values2':value_con.values})
feature  = pd.merge(feature,value_con,on='index',how='left')
feature = feature.fillna(0)
plt.figure('creativeID')
feature['values'] = feature['values2']/feature['values1']
plt.plot(feature.index,feature['values2']/feature['values1'],'.')

#---------------------camagaingnID plot -----------------------

feature = feature2.camgaignID.value_counts()
value_con = feature1.camgaignID.value_counts()
feature = pd.DataFrame({'index':feature.index.to_series().values,'values1':feature.values})
value_con = pd.DataFrame({'index':value_con.index.to_series().values,'values2':value_con.values})
feature  = pd.merge(feature,value_con,on='index',how='left')
feature = feature.fillna(0)
plt.figure('camgaingnID')
plt.plot(feature.index,feature['values2']/feature['values1'],'.')

#---------------------appID plot -----------------------
feature = feature2.appID.value_counts()
value_con = feature1.appID.value_counts()
feature = pd.DataFrame({'index':feature.index.to_series().values,'values1':feature.values})
value_con = pd.DataFrame({'index':value_con.index.to_series().values,'values2':value_con.values})
feature  = pd.merge(feature,value_con,on='index',how='left')
feature = feature.fillna(0)
plt.figure('appID')
plt.plot(feature.index,feature['values2']/feature['values1'],'.')