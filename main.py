import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
import sys,os,re
from sklearn.model_selection import KFold,StratifiedKFold
from utils import *
if False:
    import data

conf = Conf()
root = conf.root
id_name = conf.id_name
label_name = conf.label_name

def Lgb_folds_pred(nSplits=5,catFeatures=[],groups=None,swapTargetFrac=0.0,seed=42):
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
    lgb_params = {'num_leaves': 333,
          'min_child_weight': 0.03454472573214212,
          'feature_fraction': 0.3797454081646243,
          'bagging_fraction': 0.4181193142567742,
          'min_data_in_leaf': 106,
          'objective': 'binary',
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
    model.Train(trainDf=train_df,testDf=test_df,catFeatures=catFeatures,groups=groups,prefix='')
    return None

def Cab_folds_pred(nSplits=5,catFeatures=[],swapTargetFrac=0.0,seed=42):
    import cab
    random_state = seed
    cab_params = {
                'n_estimators':5000,
                'learning_rate': 0.04,
                'eval_metric':'AUC',
                'loss_function':'Logloss',
                'num_leaves': 333,
                'min_data_in_leaf': 106,
                'random_seed':seed,
                'od_wait':500,
                'task_type':'GPU',
                'depth': 8,
                #'colsample_bylevel':0.7,
                }
    model = cab.Model(params=cab_params,earlyStoppingRounds=500,verbose=500,nSplits=nSplits,swapTargetFrac=swapTargetFrac,seed=seed)
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
train_k = 1
nrows=None
train_df = pd.read_csv('%s/data/new_train_k=%s.csv'%(root,train_k),nrows=nrows)
test_df = pd.read_csv('%s/data/new_test_k=%s.csv'%(root,train_k),nrows=nrows)



for prefix in ['encoding']:#,'valueCount'
    sub_train_df = pd.read_csv('%s/data/%sTrain_k=%s.csv'%(root,prefix,train_k),nrows=nrows)
    sub_test_df = pd.read_csv('%s/data/%sTest_k=%s.csv'%(root,prefix,train_k),nrows=nrows)
    train_df = train_df.merge(sub_train_df,how='left',on=id_name)
    test_df = test_df.merge(sub_test_df,how='left',on=id_name)
    del sub_train_df,sub_test_df

cat_features = ['uniqueCard2_V258Mean']
for col in cat_features:
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)
if False:
    tmp = pd.read_csv('%s/data/train_transaction.csv'%root,usecols=['TransactionDT','D1'],nrows=nrows)
    groups = ((tmp['TransactionDT'] // 86400) - tmp['D1']).fillna(-999).values
    Lgb_folds_pred(groups=groups)
else:
    #Lgb_folds_pred()
    Cab_folds_pred(catFeatures=cat_features)
