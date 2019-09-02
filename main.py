import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
import sys,os
from sklearn.model_selection import KFold,StratifiedKFold
from utils import *

conf = Conf()
root = conf.root
id_name = conf.id_name
label_name = conf.label_name

def Lgb_folds_pred(nSplits=5,catFeatures=[],swapTargetFrac=0.0,seed=42):
    import lgb
    random_state= seed
    lgb_params1 = {
                  'objective' : 'binary',
                  'metric' : 'auc',
                  'boosting': 'gbdt',
                  'max_depth' : 11,
                  'num_leaves' : 63,
                  'learning_rate' : 0.05,
                  'bagging_freq': 5,
                  'bagging_fraction' : 0.7,
                  'feature_fraction' : 0.8,
                  'min_data_in_leaf': 80,
                  #'min_sum_heassian_in_leaf': 10,
                  'tree_learner': 'serial',
                  'boost_from_average': 'false',
                  #'lambda_l1' : 5,
                  'lambda_l2' : 1,
                  'bagging_seed' : random_state,
                  'verbosity' : 1,
                  'seed': random_state
    }
    lgb_params = {'num_leaves': 491,
          'min_child_weight': 0.03454472573214212,
          'feature_fraction': 0.3797454081646243,
          'bagging_fraction': 0.4181193142567742,
          'min_data_in_leaf': 106,
          'objective': 'binary',
          'metric':'auc',
          'max_depth': -1,
          'learning_rate': 0.006883242363721497,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.3899927210061127,
          'reg_lambda': 0.6485237330340494,
          'random_state': 47
         }
    lgb_params2 = {
                    'objective':'binary',
                    'boosting_type':'gbdt',
                    'metric':'auc',
                    'n_jobs':-1,
                    'num_leaves' : 255,
                    'learning_rate':0.01,
                    'max_depth':-1,
                    'colsample_bytree': 0.85,
                    'subsample':0.85,
                    'reg_alpha': 0.15,
                    'reg_lambda': 0.85,
                    'verbose':-1,
                    'seed': 0,
                }
    model = lgb.Model(params=lgb_params,rounds=4000,earlyStoppingRounds=200,verbose=100,nSplits=nSplits,swapTargetFrac=swapTargetFrac,seed=seed)
    model.Train(trainDf=train_df,testDf=test_df,catFeatures=catFeatures,prefix='')
    return None

def Cab_folds_pred(nSplits=5,catFeatures=[],swapTargetFrac=0.0,seed=42):
    import cab
    random_state = seed
    cab_params = {
                  'objective' : 'Logloss',
                  'eval_metric' : 'F1',
                  'num_boost_round': 3000,
                  'max_depth' : 11,
                  'num_leaves' : 63,
                  'learning_rate' : 0.05,
                  #'subsample' : 0.7,
                  #'colsample_bylevel' : 0.8,
                  'min_data_in_leaf': 80,
                  'l2_leaf_reg' : 1,
                  'task_type':'GPU',
                  'verbose' : True,
                  'random_seed': random_state
    }
    model = cab.Model(params=cab_params,earlyStoppingRounds=200,verbose=100,nSplits=nSplits,swapTargetFrac=swapTargetFrac,seed=seed)
    model.Train(trainDf=train_df,testDf=test_df,catFeatures=catFeatures,prefix='')
    return None

card_cols = ['card%s'%(i+1) for i in range(6)]
addr_cols = ['addr1','addr2']
email_cols = ['P_emaildomain','R_emaildomain']
c_cols = ['C%s'%(i+1) for i in range(14)]
d_cols = ['D%s'%(i+1) for i in range(15)]
m_cols = ['M%s'%(i+1) for i in range(9)]
v_cols = ['V%s'%(i+1) for i in range(339)]
id_cols = ['id_%s'%str(i+1).zfill(2) for i in range(38)]
device_cols = ['DeviceType','DeviceInfo']
nrows=None
train_df = pd.read_csv('%s/data/new_train.csv'%root,nrows=nrows)
test_df = pd.read_csv('%s/data/new_test.csv'%root,nrows=nrows)
for prefix in ['uniqueCrad','encoding']:#,'valueCount'
    sub_train_df = pd.read_csv('%s/data/%sTrain.csv'%(root,prefix),nrows=nrows)
    sub_test_df = pd.read_csv('%s/data/%sTest.csv'%(root,prefix),nrows=nrows)
    train_df = train_df.merge(sub_train_df,how='left',on=id_name)
    test_df = test_df.merge(sub_test_df,how='left',on=id_name)
    del sub_train_df,sub_test_df
tt_df = train_df.append(test_df)
drop_cols = ['V%sCount'%(i+1) for i in range(339)]#['TransactionDT_600', 'TransactionDT_1800', 'TransactionDT_3600', 'TransactionDT_7200', 'TransactionDT_18000','nan-449562-369913.1'] + [col for col in train_df.columns if 'CumTarget' in col]#['day_V%sDivMean'%(i+1) for i in range(339)] + ['C11DivC%s'%(i+1) for i in range(10)]
#print(drop_cols)
tt_df = tt_df.drop(drop_cols,axis=1)
cols = [col for col in tt_df.columns if 'TranDist' in col]
tt_df = Count_label_encoding(tt_df,cols)
train_df = tt_df[:train_df.shape[0]]
test_df = tt_df[train_df.shape[0]:]
'''
v_train_df = pd.read_csv('%s/data/train_transaction.csv'%root,nrows=nrows)
v_test_df = pd.read_csv('%s/data/test_transaction.csv'%root,nrows=nrows)
train_df = train_df.merge(v_train_df[[id_name]+v_cols],how='left',on=id_name)
test_df = test_df.merge(v_test_df[[id_name]+v_cols],how='left',on=id_name)
'''
del tt_df
Lgb_folds_pred()
#Cab_folds_pred(catFeatures=cat_cols)
