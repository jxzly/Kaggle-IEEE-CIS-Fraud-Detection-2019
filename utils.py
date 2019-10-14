import datetime
from contextlib import contextmanager
import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score

id_name = 'TransactionID'
label_name = 'isFraud'

class Conf():
    def __init__(self):
        self.c = 'Kaggle-IEEE-CIS-Fraud-Detection-2019'
        self.root = '/'.join(os.path.realpath(__file__).split('/')[:-1])#if your system is windows, you should manually modify the root path
        self.id_name = id_name
        self.label_name = label_name
        self.random_seed = 47
        for path in ['log','submission','valid','model']:
            if not os.path.exists('%s/%s'%(self.root,path)):
                os.mkdir('%s/%s'%(self.root,path))

@contextmanager
def Timer(title):
    'timing function'
    t0 = datetime.datetime.now()
    yield
    print("%s - done in %is"%(title, (datetime.datetime.now() - t0).seconds))
    return None

def Metric(target,pred):
    return roc_auc_score(target,pred)

def Write_log(logFile,text,isPrint=True):
    if isPrint:
        print(text)
    logFile.write(text)
    logFile.write('\n')
    return None

def Information_entropy(x):
    x = pd.Series(x)
    pct = x.groupby(by=x).count() / len(x)
    return np.sum(-pct*np.log2(pct))

def Count_encoding(df,cols,drop=True):
    for col in cols:
        print('count encoding:',col)
        df['%sCount'%col] = df[col].map(dict(df[col].value_counts()))
    if drop:
        df.drop(cols,axis=1,inplace=True)
    return df

def Count_label_encoding(df,cols,sparseThreshold=50):
    for col in cols:
        print('count label encoding:',col)
        df['tmp'] = df[col].map(dict(df[col].value_counts())).astype(int)
        df.loc[df['tmp']<sparseThreshold,col] = df.loc[df['tmp']<sparseThreshold,'tmp'].astype(str)
        df.loc[df['tmp']>=sparseThreshold,col] = df.loc[df['tmp']>=sparseThreshold,'tmp'].astype(str) + '_' + df.loc[df['tmp']>=sparseThreshold,col].astype(str)
        df[col] = df[col].rank(method='dense')
    df.drop(['tmp'],axis=1,inplace=True)
    return df

def Mean_encoding(df,encodingCols,cols,drop=True):
    encoding_cols = []
    for encoding_col in encodingCols:
        for col in cols:
            print('mean encoding %s by %s'%(encoding_col,col))
            encoding_cols.append('%s_%sMean'%(encoding_col,col))
            df['%s_%sMean'%(encoding_col,col)] = df[[encoding_col,col]].groupby([encoding_col])[col].transform('mean')
    if drop:
        df.drop(encodingCols,axis=1,inplace=True)
    return encoding_cols,df

def Std_encoding(df,encodingCols,cols,drop=True):
    encoding_cols = []
    for encoding_col in encodingCols:
        for col in cols:
            print('std encoding %s by %s'%(encoding_col,col))
            encoding_cols.append('%s_%sMean'%(encoding_col,col))
            df['%s_%sStd'%(encoding_col,col)] = df[[encoding_col,col]].groupby([encoding_col])[col].transform('std')
    if drop:
        df.drop(encodingCols,axis=1,inplace=True)
    return encoding_cols,df

def Skew_encoding(df,encodingCols,cols,drop=True):
    encoding_cols = []
    for encoding_col in encodingCols:
        for col in cols:
            print('skew encoding %s by %s'%(encoding_col,col))
            encoding_cols.append('%s_%sMean'%(encoding_col,col))
            df['%s_%sSkew'%(encoding_col,col)] = df[[encoding_col,col]].groupby([encoding_col])[col].transform('skew')
    if drop:
        df.drop(encodingCols,axis=1,inplace=True)
    return encoding_cols,df

def Norm(x):
    x = np.array(x)
    if len(x) == 1:
        return [0]
    if len(x[~np.isnan(x)]) == 0:
        return x
    x[np.isnan(x)] = np.mean(x[~np.isnan(x)])
    if x.min() == x.max():
        return [-1 for i in range(len(x))]
    return (x-x.min())/(x.max()-x.min())

if __name__ == '__main__':
    df = pd.DataFrame({'a':['1','2_as','2_as','2_ff','2_ff','2_as','2_ff','1']})
    print(Count_encoding(df,['a'],3))
