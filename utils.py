import datetime
from contextlib import contextmanager
import pandas as pd
import numpy as np
import os
from sklearn.metrics import f1_score,roc_auc_score

id_name = 'TransactionID'
label_name = 'isFraud'

class Conf():
    def __init__(self):
        self.c = 'Kaggle-IEEE-CIS-Fraud-Detection-2019'
        self.root = '/'.join(os.path.realpath(__file__).split('/')[:-1])#'F:/home/daishu/competition/%s/'%self.c#
        self.id_name = id_name
        self.label_name = label_name
        self.random_state = 47

@contextmanager
def Timer(title):
    '对运行部分进行计时'
    t0 = datetime.datetime.now()
    yield
    print("%s - done in %is"%(title, (datetime.datetime.now() - t0).seconds))
    return None

def Get_file_name(filePath):
    '获取目录下的文件名'
    for root, dires, files in os.walk(filePath):
        return files

def Metric(target,pred):
    return roc_auc_score(target,pred)

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

'''
def Count_encoding(df,cols,sparseThreshold=50):
    for col in cols:
        print(col)
        df['tmp'] = df[col].map(dict(df[col].value_counts())).astype(int)
        df.loc[df['tmp']<sparseThreshold,col] = df.loc[df['tmp']<sparseThreshold,'tmp'].astype(str)
        df.loc[df['tmp']>=sparseThreshold,col] = df.loc[df['tmp']>=sparseThreshold,'tmp'].astype(str) + '_' + df.loc[df['tmp']>=sparseThreshold,col].astype(str)
        df[col] = df[col].rank(method='dense')
    df.drop(['tmp'],axis=1,inplace=True)
    return df
'''
def Count_encoding(df,cols,drop=True):
    for col in cols:
        print('count encoding:',col)
        df['%sCount'%col] = df[col].map(dict(df[col].value_counts())).astype(int)
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

def Target_encoding(df,cols,trainNrows,sparseThreshold=50):
    target_mean = df[:trainNrows][label_name].mean()
    for col in cols:
        print('target encoding:',col)
        df['tmp'] = target_mean
        train_value_counts = df[:trainNrows][col].value_counts()
        gt_sparse_threshold_value = train_value_counts[train_value_counts>sparseThreshold].index
        df.loc[df[col].isin(gt_sparse_threshold_value),'tmp'] = df.loc[df[col].isin(gt_sparse_threshold_value),col].map(dict(df[:trainNrows].groupby([col])[label_name].mean()))
        df[col] = df['tmp']
    df.drop(['tmp'],axis=1,inplace=True)
    return df

def Mean_encoding(df,encodingCol,cols,drop=True):
    encoding_cols = []
    for col in cols:
        print('mean encoding:',col)
        encoding_cols.append('%s_%sMean'%(encodingCol,col2))
        df['%s_%sMean'%(encodingCol,col2)] = df[[encodingCol,col]].groupby([encodingCol])[col].transform('mean')
    if drop:
        df.drop([encodingCol],axis=1,inplace=True)
    return encoding_cols,df

def Get_list_ave_interval(x):
    if len(x) < 2:
        return -999
    else:
        return np.mean(np.diff(x))

def Get_count_sum(x,shift=0):
    x = list(range(shift,len(x)+shift))
    if shift != 0:
        x = np.roll(x,shift)
        for i in range(shift):
            x[i] = 0
    return x

def Get_cum_sum(x,shift=0):
    if shift != 0:
        x = np.roll(x,shift)
        for i in range(shift):
            x[i] = 0
    return np.cumsum(x)

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
    df = pd.DataFrame({'a':[1,2,3,4,4,4],'isFraud':[0,0,1,0,1,1]})
    print(Target_encoding(df,['a'],4,1))
