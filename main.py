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
    lgb_params = {
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

train_df = pd.read_csv('%s/data/new_train.csv'%root,nrows=None)
test_df = pd.read_csv('%s/data/new_test.csv'%root,nrows=None)
Lgb_folds_pred()
#Cab_folds_pred(catFeatures=cat_cols)
