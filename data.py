import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
import datetime
import os,sys
from multiprocessing import Process,Manager
from collections import ChainMap
from utils import *

conf = Conf()
root = conf.root
id_name = conf.id_name
label_name = conf.label_name

def Get_train_test():
    train_df1 = pd.read_csv('%s/data/train_identity.csv'%root)
    train_df2 = pd.read_csv('%s/data/train_transaction.csv'%root)
    test_df1 = pd.read_csv('%s/data/test_identity.csv'%root)
    test_df2 = pd.read_csv('%s/data/test_transaction.csv'%root)
    train_df = train_df2.merge(train_df1,how='left',on=id_name)
    test_df = test_df2.merge(test_df1,how='left',on=id_name)
    cat_cols = ['card%s'%(i+1) for i in range(6)]
    cat_cols.extend(['TransactionAmt','ProductCD','addr1','addr2'])
    for col in cat_cols:
        train_df[col] = train_df[col].astype(str).apply(lambda x:x.replace('.0',''))
        test_df[col] = test_df[col].astype(str).apply(lambda x:x.replace('.0',''))
    return train_df,test_df

def Get_tran_features(df):
    times = [300,1800]
    for t in times:
        df['tranNumFirst%s'%t] = 0
        df['tranNumLater%s'%t] = 0
        df['tranNumIn%s'%t] = 0
    df['lastTranTD'] = 0
    df['nextTranTD'] = 0
    for c in df['card'].unique():
        tmp = df.loc[df['card']==c]
        tmp['addr1_0'] = np.roll(tmp['addr1'],1)
        tmp.loc[tmp.index[0],'addr1_0'] = 'nan'
        tmp['addr1_1'] = tmp['addr1']
        tmp['addr1_2'] = np.roll(tmp['addr1'],-1)
        tmp.loc[tmp.index[-1],'addr1_0'] = 'nan'
        tmp['addr2_0'] = np.roll(tmp['addr2'],1)
        tmp.loc[tmp.index[0],'addr2_0'] = 'nan'
        tmp['addr2_1'] = tmp['addr2']
        tmp['addr2_2'] = np.roll(tmp['addr2'],-1)
        tmp.loc[tmp.index[-1],'addr2_0'] = 'nan'
        diff = tmp['TransactionDT'].diff().fillna(99999999)
        df.loc[tmp.index,'lastTranTD'] = diff
        df.loc[tmp.index,'nextTranTD'] = np.roll(diff,-1)
        df.loc[tmp.index,'lastTranDist'] = tmp[['addr1_0','addr2_0','addr1_1','addr1_1']].apply(lambda x:'_'.join(x),axis=1)
        df.loc[tmp.index,'nextTranDist'] = tmp[['addr1_1','addr2_1','addr1_2','addr1_2']].apply(lambda x:'_'.join(x),axis=1)
        for ind in tmp.index:
            tran_t = tmp.loc[ind,'TransactionDT']
            for t in times:
                df.loc[ind,'tranNumFirst%s'%t] = tmp.loc[(tmp['TransactionDT']<=tran_t)&(tmp['TransactionDT']>=tran_t-t)].shape[0]
                df.loc[ind,'tranNumLater%s'%t] = tmp.loc[(tmp['TransactionDT']>=tran_t)&(tmp['TransactionDT']<=tran_t+t)].shape[0]
                df.loc[ind,'tranNumIn%s'%t] = tmp.loc[(tmp['TransactionDT']>=tran_t-t//2)&(tmp['TransactionDT']<=tran_t+t//2)].shape[0]
    return df

def Get_card_id_features(df,cardInfo,prefix):
    #card_info = ['card%s'%(i+1) for i in range(6)]
    #card_info.extend(['TransactionAmt','ProductCD'])
    df[prefix] = df[card_info].apply(lambda x:'_'.join(x),axis=1)
    tran_df = Get_tran_features(df[[prefix,'TransactionDT','addr1','addr2']]).add_prefix(prefix+'_')
    df = pd.concat([df,tran_df.drop([prefix,'TransactionDT','addr1','addr2'],axis=1)],axis=1)
    df[prefix] = df['TransactionAmt'] / df[prefix].map(dict(df[prefix].value_counts()))
    df[prefix] = Count_encoding(df[[prefix]])[prefix]

    return df
