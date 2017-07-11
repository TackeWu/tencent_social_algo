import pandas as pd
import numpy as np
import time
from joblib import dump,load


def con_tables(train,test,table_name,con_col):
    start = time.time()
    print("-------------->读取"+table_name+".csv")
    table = pd.read_csv('./data/'+table_name+'.csv')
    print("读取用时",time.time()-start)
    print("-------------->merge train、test "+table_name)
    train = pd.merge(train, table, on=con_col, sort=False, how='left')
    test = pd.merge(test, table, on=con_col, sort=False, how='left')

    return train,test

if __name__ == "__main__":
    print("*****************读取train、test***********************")
    start = time.time()
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')

    table_name = ['ad','user','position','app_categories']
    con_col =['creativeID','userID','positionID','appID']

    for i in range(len(table_name)):
        train,test = con_tables(train,test,table_name[i],con_col[i])

    print("*****************存储train、test***********************")
    start = time.time()

    dump(train, "./dump/train.joblib_dat")
    dump(test, "./dump/test.joblib_dat")
    print("存储用时", time.time() - start)

