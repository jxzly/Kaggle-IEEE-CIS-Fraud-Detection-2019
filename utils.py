import datetime
from contextlib import contextmanager
import pandas as pd
import numpy as np
import os
from sklearn.metrics import f1_score,roc_auc_score

class Conf():
    def __init__(self):
        self.c = 'Kaggle-IEEE-CIS-Fraud-Detection-2019'
        self.root = '/'.join(os.path.realpath(__file__).split('/')[:-1])#'F:/home/daishu/competition/%s/'%self.c#
        self.id_name = 'TransactionID'
        self.label_name = 'isFraud'

@contextmanager
def Timer(title):
    '对运行部分进行计时'
    t0 = datetime.datetime.now()
    yield
    print("%s - done in %is"%(title, (datetime.datetime.now() - t0).seconds))
    return None

def AUC(target,pre):
    return roc_auc_score(target,pre)

def Lgb_f1_score(preds, data):
    labels = data.get_label()
    score = f1_score(labels, np.round(preds))
    return 'f1', score, True

def Write_log(logFile,text,isPrint=True):
    if isPrint:
        print(text)
    logFile.write(text)
    logFile.write('\n')
    return None

def Reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                c_prec = df[col].apply(lambda x: np.finfo(x).precision).max()
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max and c_prec == np.finfo(np.float32).precision:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def Information_entropy(x):
    x = pd.Series(x)
    pct = x.groupby(by=x).count() / len(x)
    return np.sum(-pct*np.log2(pct))

def Mutual_information(x,y):
    df = pd.DataFrame({'x':x,'y':y})
    entropy = Information_entropy(y)
    conditiona_entropy = df.groupby(['x']).agg({'y':Information_entropy})
    pct = df.groupby(['x']).count() / len(x)
    return entropy - np.sum(pct.values*conditiona_entropy.values)

def Data_review(trainDf,testDf,idName,labelName):
    info = []
    info.append([idName,'object',trainDf.shape[0],testDf.shape[0],0,'0.00%',np.nan,np.nan,np.nan,np.nan,np.nan])
    info.append([labelName,trainDf[labelName].dtype,trainDf.shape[0],testDf.shape[0],np.nan,np.nan,trainDf[labelName].mean(),np.nan,np.nan,np.nan,np.nan])
    for col in tqdm(trainDf.columns):
        if col in [idName,labelName]:
            continue
        data_type = trainDf[col].dtype
        train_nunique = trainDf[col].fillna('-999').nunique()
        test_nunique = testDf[col].fillna('-999').nunique()
        train_test_intersection = len(set(trainDf[col].unique())&(set(testDf[col].unique())))
        intersectiom_pct = '%.2f%%'%(train_test_intersection/train_nunique*100)
        if data_type == 'object':
            train_mode_or_mean = trainDf[col].mode().values[0]
            test_mode_or_mean = testDf[col].mode().values[0]
            mi_or_corr = Mutual_information(trainDf[col],trainDf[labelName])
        else:
            train_mode_or_mean = trainDf[col].mean()
            test_mode_or_mean = testDf[col].mean()
            mi_or_corr = trainDf[col].corr(trainDf[labelName])
        train_nan_num = trainDf[col].isnull().sum()
        test_nan_num = testDf[col].isnull().sum()
        info.append([col,data_type,train_nunique,test_nunique,train_test_intersection,intersectiom_pct,train_mode_or_mean,test_mode_or_mean,train_nan_num,test_nan_num,mi_or_corr])
    info_df = pd.DataFrame(info,columns=['column','dataType','trainNunique','testNunique','trainTestIntersection','intersectionPct','trainModeOrMean','testModeOrMean','trainNanNum','testNanNum','miOrCorr'])
    return info_df

def Count_encoding(df,cols,sparseThreshold=50):
    for col in cols:
        print(col)
        df['tmp'] = df[col].map(dict(df[col].value_counts())).astype(int)
        df.loc[df['tmp']<sparseThreshold,col] = df.loc[df['tmp']<sparseThreshold,'tmp'].astype(str)
        df.loc[df['tmp']>=sparseThreshold,col] = df.loc[df['tmp']>=sparseThreshold,'tmp'].astype(str) + '_' + df.loc[df['tmp']>=sparseThreshold,col].astype(str)
        df[col] = df[col].rank(method='dense')
    df.drop(['tmp'],axis=1,inplace=True)
    return df

if __name__ == '__main__':
    df = pd.DataFrame({'a':['1','2_as','2_as','2_ff','2_ff','2_as','2_ff','1']})
    print(Count_encoding(df,['a'],3))
