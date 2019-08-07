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

np.random.seed(2019)

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
                df[col] = df[col].fillna('-999')
            else:
                df[col] = df[col].fillna(-999)
    cat_cols = ['card%s'%(i+1) for i in range(6)]
    cat_cols.extend(['TransactionAmt','ProductCD','addr1','addr2'])
    for col in cat_cols:
        df[col] = df[col].astype(str).apply(lambda x:x.replace('.0',''))
    return df

def Get_tran_features(df,prefix,resultList):
    '''
    times = [300,1800]
    for t in times:
        df['tranNumFirst%s'%t] = 0
        df['tranNumLater%s'%t] = 0
        df['tranNumIn%s'%t] = 0
    '''
    df['lastTranTD'] = 0
    df['nextTranTD'] = 0
    df['lastTranDist'] = ''
    df['nextTranDist'] = ''
    with Timer('test1'):
        for c in df[prefix].unique():
            tmp = df.loc[df[prefix]==c]
            tmp['addr1_0'] = np.roll(tmp['addr1'],1)
            tmp.loc[tmp.index[0],'addr1_0'] = '-999'
            tmp['addr1_1'] = tmp['addr1']
            tmp['addr1_2'] = np.roll(tmp['addr1'],-1)
            tmp.loc[tmp.index[-1],'addr1_2'] = '-999'
            tmp['addr2_0'] = np.roll(tmp['addr2'],1)
            tmp.loc[tmp.index[0],'addr2_0'] = '-999'
            tmp['addr2_1'] = tmp['addr2']
            tmp['addr2_2'] = np.roll(tmp['addr2'],-1)
            tmp.loc[tmp.index[-1],'addr2_2'] = '-999'
            diff = tmp['TransactionDT'].diff().fillna(99999999)
            df.loc[tmp.index,'lastTranTD'] = diff
            df.loc[tmp.index,'nextTranTD'] = np.roll(diff,-1)
            df.loc[tmp.index,'lastTranDist'] = tmp[['addr1_0','addr2_0','addr1_1','addr2_1']].apply(lambda x:'_'.join(x),axis=1)
            df.loc[tmp.index,'nextTranDist'] = tmp[['addr1_1','addr2_1','addr1_2','addr2_2']].apply(lambda x:'_'.join(x),axis=1)
            '''
            with Timer('test2'):
                for ind in tmp.index:
                    tran_t = tmp.loc[ind,'TransactionDT']
                    for t in times:
                        df.loc[ind,'tranNumFirst%s'%t] = tmp.loc[(tmp['TransactionDT']<=tran_t)&(tmp['TransactionDT']>=tran_t-t)].shape[0]
                        df.loc[ind,'tranNumLater%s'%t] = tmp.loc[(tmp['TransactionDT']>=tran_t)&(tmp['TransactionDT']<=tran_t+t)].shape[0]
                        df.loc[ind,'tranNumIn%s'%t] = tmp.loc[(tmp['TransactionDT']>=tran_t-t//2)&(tmp['TransactionDT']<=tran_t+t//2)].shape[0]
            '''
    resultList.append(df)
    return None

def Get_card_id_features(df,cardInfo,prefix,nJobs=8):
    #card_info = ['card%s'%(i+1) for i in range(6)]
    #card_info.extend(['TransactionAmt','ProductCD'])
    df[prefix] = df[cardInfo].apply(lambda x:'_'.join(x),axis=1)
    tmp = df[[prefix]]
    tmp[prefix] = tmp[prefix].map(dict(tmp[prefix].value_counts()))
    unique_card = list(df.loc[tmp[prefix]>1,prefix].unique())
    np.random.shuffle(unique_card)
    batch_size = int(np.ceil(len(unique_card)/nJobs))
    result_list = Manager().list()
    jobs = []
    for i in range(nJobs):
        sub_df = df.loc[df[prefix].isin(unique_card[i*batch_size:(i+1)*batch_size]),[prefix,'TransactionDT','addr1','addr2']]
        print('-%s job df.shape:%s'%(i,sub_df.shape))
        jobs.append(Process(target=Get_tran_features,args=(sub_df,prefix,result_list,)))
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()
    tran_df = pd.concat(result_list).sort_index().drop([prefix,'TransactionDT','addr1','addr2'],axis=1)#Get_tran_features(df[[prefix,'TransactionDT','addr1','addr2']],prefix).drop([prefix,'TransactionDT','addr1','addr2'],axis=1)
    tran_df = tran_df.add_prefix(prefix+'_')
    df = pd.concat([df,tran_df],axis=1)
    df.loc[tmp[prefix]==1,prefix+'_lastTranTD'] = 9999999
    df.loc[tmp[prefix]==1,prefix+'_nextTranTD'] = 9999999
    df.loc[tmp[prefix]==1,prefix+'_lastTranDist'] = df.loc[tmp[prefix]==1,['addr1','addr2']].apply(lambda x:'-999_-999_'+'_'.join(x),axis=1)
    df.loc[tmp[prefix]==1,prefix+'_nextTranDist'] = df.loc[tmp[prefix]==1,['addr1','addr2']].apply(lambda x:'_'.join(x)+'_-999_-999',axis=1)
    #df['%s_amtDivCount'%prefix] = df['TransactionAmt'].astype(float) / df[prefix].map(dict(df[prefix].value_counts()))
    #df = Count_encoding(df,[prefix])
    return df

def Get_tt_card_id_features(df,cardInfo,prefix):
    df['%s_amtDivCount'%prefix] = df['TransactionAmt'].astype(float) / df[prefix].map(dict(df[prefix].value_counts()))
    df = Count_encoding(df,[prefix,prefix+'_lastTranDist',prefix+'_nextTranDist'])
    return df

def Get_t_features(df):
    df['dayOfWeek'] = np.floor(df['TransactionDT']/(3600*24)) % 7
    df['hour'] = hours = np.floor(df['TransactionDT']/3600) % 24
    df.drop(['TransactionDT'],axis=1,inplace=True)
    return df

train_df,test_df = Get_train_test(nrows=None)
train_nrows = train_df.shape[0]
train_df = Get_nan_features(train_df)
test_df = Get_nan_features(test_df)
train_df = Get_card_id_features(train_df,card_cols,'uniqueCrad0')
test_df = Get_card_id_features(test_df,card_cols,'uniqueCrad0')
tt_df = train_df.append(test_df).reset_index(drop=True)
del train_df,test_df
tt_df = Get_tt_card_id_features(tt_df,card_cols,'uniqueCrad0')
tt_df = Get_t_features(tt_df)
tt_df = Count_encoding(tt_df,card_cols+email_cols+m_cols+id_cols+device_cols+['ProductCD'])
print(tt_df.head())
tt_df[:train_nrows].to_csv('%s/data/new_train.csv'%root,index=False)
tt_df[train_nrows:].to_csv('%s/data/new_test.csv'%root,index=False)
