import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
import sys,os,re
from sklearn.model_selection import KFold,StratifiedKFold
from utils import *

conf = Conf()
root = conf.root
id_name = conf.id_name
label_name = conf.label_name
random_seed = conf.random_seed

def Lgb_folds_pred(nSplits=5,catFeatures=[],groups=None):
    import lgb
    random_state= seed
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
    model = lgb.Model(params=lgb_params,rounds=4000,earlyStoppingRounds=200,verbose=100,nSplits=nSplits,seed=random_seed)
    model.Train(trainDf=train_df,testDf=test_df,catFeatures=catFeatures,groups=groups,prefix='train_k>%s_'%train_k)
    return None

for train_k in [-1,0,1]:
    nrows=None
    train_df = pd.read_csv('%s/data/new_train_k>%s.csv'%(root,train_k),nrows=nrows)
    test_df = pd.read_csv('%s/data/new_test_k>%s.csv'%(root,train_k),nrows=nrows)
    for prefix in ['encoding']:
        sub_train_df = pd.read_csv('%s/data/%sTrain.csv'%(root,prefix),nrows=nrows)
        sub_test_df = pd.read_csv('%s/data/%sTest.csv'%(root,prefix),nrows=nrows)
        train_df = train_df.merge(sub_train_df,how='left',on=id_name)
        test_df = test_df.merge(sub_test_df,how='left',on=id_name)
        del sub_train_df,sub_test_df
    Lgb_folds_pred()
    del train_df,test_df
