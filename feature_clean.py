import os,sys
import lightgbm as lgb
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold,StratifiedKFold
from tqdm import tqdm
from utils import *

conf = Conf()
root = conf.root
id_name = conf.id_name
label_name = conf.label_name
random_state = conf.random_state

def Look_for_diff_distribution_features(trainDf,lgbParams,verbose=50,shuffle=False):
    log = open('%s/featureClean/diff/log/train.log'%root,'w')
    train_cols = [col for col in trainDf.columns if col not in [id_name,'isTrain']]
    Write_log(log,str(train_cols))
    Write_log(log,str(lgbParams))
    y = trainDf['isTrain'].values
    copy_y = copy.deepcopy(y)
    if shuffle:
        prefix = 'shuffle'
        np.random.shuffle(copy_y)
    else:
        prefix = 'actual'
    evals_result_dic = {}
    train_data = lgb.Dataset(trainDf[train_cols], copy_y,silent=1)
    model = lgb.train(lgbParams,
        train_data,
        1000,
        valid_sets=train_data,
        evals_result = evals_result_dic,
        verbose_eval=50
    )
    print(evals_result_dic)
    plt.plot(evals_result_dic['training'][lgbParams['metric']])
    train_pre = model.predict(trainDf[train_cols])
    train_metric = Metric(copy_y,train_pre)
    print('训练样本 metric:%.8f'%(train_metric))
    for i in range(len(evals_result_dic['training'][lgbParams['metric']])//verbose):
        Write_log(log,' - %i round - train_metric: %.6f\n'%(i*verbose,evals_result_dic['training'][lgbParams['metric']][i*verbose]))
    importance_gain = model.feature_importance(importance_type='gain')
    importance_split = model.feature_importance(importance_type='split')
    feature_name = model.feature_name()
    feature_importance_df = pd.DataFrame({'feature_name':feature_name,'importance_gain':importance_gain,'importance_split':importance_split} )
    feature_importance_df = feature_importance_df.sort_values(by=['importance_gain'],ascending=False)
    feature_importance_df.to_csv('%s/featureClean/diff/valid/%s_feature_importance_metric_%.8f.csv'%(root,prefix,train_metric),index=False)
    plt.savefig('%s/featureClean/diff/log/%s_train_lgb_metric_%.8f.png'%(root,prefix,train_metric))
    plt.close('all')
    log.close()
    os.rename('%s/featureClean/diff/log/train.log'%root, '%s/featureClean/diff/log/%s_train_lgb_metric_%.8f.log'%(root,prefix,train_metric))
    return None

def Get_diff_features(threshold=0):
    feature_importance_files = Get_file_name('%s/featureClean/diff/valid/'%root)
    actual_feature_importance_df = pd.DataFrame()
    fake_feature_importance_df = pd.DataFrame()
    for file in feature_importance_files:
        if 'shuffle' not in file:
            actual_feature_importance_df = actual_feature_importance_df.append(pd.read_csv('%s/featureClean/diff/valid/%s'%(root,file)))
        else:
            fake_feature_importance_df = fake_feature_importance_df.append(pd.read_csv('%s/featureClean/diff/valid/%s'%(root,file)))
    feature_score = []
    for f in actual_feature_importance_df['feature_name'].unique():
        f_fake_feature_importance_gain = fake_feature_importance_df.loc[fake_feature_importance_df['feature_name'] == f, 'importance_gain'].values
        f_actual_feature_importance_gain = actual_feature_importance_df.loc[actual_feature_importance_df['feature_name'] == f, 'importance_gain'].mean()
        gain_score = np.log(1e-10 + f_actual_feature_importance_gain / (1 + np.percentile(f_fake_feature_importance_gain, threshold)))  # Avoid didvide by zero
        f_fake_feature_importance_split = fake_feature_importance_df.loc[fake_feature_importance_df['feature_name'] == f, 'importance_split'].values
        f_actual_feature_importance_split = actual_feature_importance_df.loc[actual_feature_importance_df['feature_name'] == f, 'importance_split'].mean()
        split_score = np.log(1e-10 + f_actual_feature_importance_split / (1 + np.percentile(f_fake_feature_importance_split, threshold)))  # Avoid didvide by zero
        feature_score.append((f, split_score, gain_score))
    score_df = pd.DataFrame(feature_score, columns=['feature_name', 'split_score', 'gain_score'])
    features = score_df.loc[(score_df['split_score']>0.0)&(score_df['gain_score']>0.0),'feature_name'].values
    print(features)
    return None

'''train_df = pd.read_csv('%s/data/new_train.csv'%root,nrows=None)
test_df = pd.read_csv('%s/data/new_test.csv'%root,nrows=None)
train_df['isTrain'] = 1
test_df['isTrain'] = 0
tt_df = train_df.append(test_df)
cols = [col for col in tt_df.columns if 'TranDist' in col]
tt_df = Count_encoding(tt_df,cols)
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
Look_for_diff_distribution_features(tt_df.drop([label_name],axis=1),lgb_params,shuffle=True)
'''
Get_diff_features()
