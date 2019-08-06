import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
import datetime
import os,sys
from multiprocessing import Process,Manager
from tqdm import tqdm
from utils import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

conf = Conf()
root = conf.root
id_name = conf.id_name
label_name = conf.label_name
card_cols = ['card%s'%(i+1) for i in range(6)]
email_cols = ['P_emaildomain','R_emaildomain']
c_cols = ['C%s'%(i+1) for i in range(14)]
d_cols = ['D%s'%(i+1) for i in range(15)]
m_cols = ['M%s'%(i+1) for i in range(9)]
v_cols = ['V%s'%(i+1) for i in range(339)]
id_cols = ['id_%s'%str(i+1).zfill(2) for i in range(38)]
device_cols = ['DeviceType','DeviceInfo']

def Get_train_test(nrows=None):
    train_df1 = pd.read_csv('%s/data/train_identity.csv'%root)
    train_df2 = pd.read_csv('%s/data/train_transaction.csv'%root,nrows=nrows)
    test_df1 = pd.read_csv('%s/data/test_identity.csv'%root)
    test_df2 = pd.read_csv('%s/data/test_transaction.csv'%root,nrows=nrows)
    train_df = train_df2.merge(train_df1,how='left',on=id_name)
    test_df = test_df2.merge(test_df1,how='left',on=id_name)
    test_df[label_name] = -1
    return train_df,test_df

def Get_nan_features(df):
    df['nanNum'] = df.isna().sum(axis=1)
    df['cardNanNum'] = df[card_cols].isna().sum(axis=1)
    df['emailNanNum'] = df[email_cols].isna().sum(axis=1)
    df['cNanNum'] = df[c_cols].isna().sum(axis=1)
    df['dNanNum'] = df[d_cols].isna().sum(axis=1)
    df['mNanNum'] = df[m_cols].isna().sum(axis=1)
    df['vNanNum'] = df[v_cols].isna().sum(axis=1)
    df['idNanNum'] = df[id_cols].isna().sum(axis=1)
    df['deviceNanNum'] = df[device_cols].isna().sum(axis=1)
    for col in df.columns:
        if col in [id_name,label_name]:
            continue
        else:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('-817119')
            else:
                df[col] = df[col].fillna(-817119)
    cat_cols = ['card%s'%(i+1) for i in range(6)]
    cat_cols.extend(['TransactionAmt','ProductCD','addr1','addr2'])
    for col in cat_cols:
        df[col] = df[col].astype(str).apply(lambda x:x.replace('.0',''))
    return df

def Get_tran_features(df,prefix):
    times = [300,1800]
    for t in times:
        df['tranNumFirst%s'%t] = 0
        df['tranNumLater%s'%t] = 0
        df['tranNumIn%s'%t] = 0
    df['lastTranTD'] = 0
    df['nextTranTD'] = 0
    with Timer('test1'):
        for c in df[prefix].unique():
            tmp = df.loc[df[prefix]==c]
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
            '''
            with Timer('test2'):
                for ind in tmp.index:
                    tran_t = tmp.loc[ind,'TransactionDT']
                    for t in times:
                        df.loc[ind,'tranNumFirst%s'%t] = tmp.loc[(tmp['TransactionDT']<=tran_t)&(tmp['TransactionDT']>=tran_t-t)].shape[0]
                        df.loc[ind,'tranNumLater%s'%t] = tmp.loc[(tmp['TransactionDT']>=tran_t)&(tmp['TransactionDT']<=tran_t+t)].shape[0]
                        df.loc[ind,'tranNumIn%s'%t] = tmp.loc[(tmp['TransactionDT']>=tran_t-t//2)&(tmp['TransactionDT']<=tran_t+t//2)].shape[0]
            '''
    return df

def Get_card_id_features(df,cardInfo,prefix):
    #card_info = ['card%s'%(i+1) for i in range(6)]
    #card_info.extend(['TransactionAmt','ProductCD'])
    df[prefix] = df[cardInfo].apply(lambda x:'_'.join(x),axis=1)
    tran_df = Get_tran_features(df[[prefix,'TransactionDT','addr1','addr2']],prefix).drop([prefix,'TransactionDT','addr1','addr2'],axis=1)
    tran_df = tran_df.add_prefix(prefix)
    df = pd.concat([df,tran_df],axis=1)
    df['%s_amtDivCount'%prefix] = df['TransactionAmt'].astype(float) / df[prefix].map(dict(df[prefix].value_counts()))
    df = Count_encoding(df,[prefix])
    return df

train_df,test_df = Get_train_test(nrows=100)
train_nrows = train_df.shape[0]
tt_df = train_df.append(test_df).reset_index(drop=True)
del train_df,test_df
tt_df = Get_nan_features(tt_df)
tt_df = Get_card_id_features(tt_df,card_cols,'uniqueCrad0')
print(tt_df.head())
tt_df[:train_nrows].to_csv('%s/data/new_train.csv'%root,index=False)
tt_df[train_nrows:].to_csv('%s/data/new_test.csv'%root,index=False)
