import os,sys
import lightgbm as lgb
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold
from tqdm import tqdm
from utils import *

conf = Conf()
root = conf.root
id_name = conf.id_name
label_name = conf.label_name

class Model():
    def __init__(self,params,rounds,earlyStoppingRounds,verbose,nSplits,swapTargetFrac,seed):
        self.params = params
        self.rounds = rounds
        self.early_stopping_rounds = earlyStoppingRounds
        self.verbose = verbose
        self.n_splits = nSplits
        self.swap_target_frac = swapTargetFrac
        self.seed = seed

    def Train(self,trainDf,testDf,catFeatures,groups=None,prefix=''):
        if groups is None:
            skf = KFold(n_splits=self.n_splits, shuffle=False, random_state=self.seed)
        #skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        else:
            skf = GroupKFold(n_splits=self.n_splits)
        train_cols = [col for col in trainDf.columns if col not in [id_name,label_name]]
        log = open('%s/log/train.log'%root,'w')
        Write_log(log,str(train_cols))
        Write_log(log,str(self.params))
        valid_df = trainDf[[id_name]]
        valid_df[label_name] = 0
        submission_df = testDf[[id_name]]
        feature_importance_df = pd.DataFrame()
        train_pred = []
        train_target = []
        all_valid_metric = []
        y_mean = np.mean(trainDf[label_name])
        for fold, (train_index, val_index) in enumerate(skf.split(trainDf,trainDf[label_name],groups)):

            train_y = copy.deepcopy(trainDf.loc[train_index,label_name].values)
            if self.swap_target_frac > 0.0:
                random_index = [np.random.randint(0,len(train_y)-1) for k in range(len(train_y))]
                random_y = train_y[random_index]
                swap_mat = np.random.binomial(1,self.swap_target_frac,len(train_y)).reshape(1,-1)[0]
                swap_y = train_y*(1.0-swap_mat) + random_y*swap_mat
            else:
                swap_y = train_y
            evals_result_dic = {}
            train_data = lgb.Dataset(trainDf.loc[train_index,train_cols], label=swap_y,categorical_feature=catFeatures)
            val_data = lgb.Dataset(trainDf.loc[val_index,train_cols], label=trainDf.loc[val_index,label_name],categorical_feature=catFeatures)
            model = lgb.train(self.params,
                train_data,
                self.rounds,
                valid_sets=[train_data,val_data],
                evals_result = evals_result_dic,
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=self.verbose
            )
            del train_data,val_data
            if not os.path.exists('%s/model/modelLgb/tmp/'%root):
                os.mkdir('%s/model/modelLgb/tmp/'%root)
            model.save_model('%s/model/modelLgb/tmp/model-fold-%s'%(root,fold))
            train_pred.extend(model.predict(trainDf.loc[train_index,train_cols]))
            train_target.extend(trainDf.loc[train_index,label_name].values)
            val_pred = model.predict(trainDf.loc[val_index,train_cols])
            valid_df.loc[val_index,label_name] = val_pred
            submission_df['fold%s'%(fold+1)] = model.predict(testDf[train_cols],
                       num_iteration=model.best_iteration)
            importance_gain = model.feature_importance(importance_type='gain')
            importance_split = model.feature_importance(importance_type='split')
            feature_name = model.feature_name()
            feature_importance = pd.DataFrame({'feature_name':feature_name,'importance_gain':importance_gain,'importance_split':importance_split} )
            feature_importance_df = feature_importance_df.append(feature_importance)
            Write_log(log,'fold %s\n'%fold)
            for i in range(len(evals_result_dic['valid_1'][self.params['metric']])//self.verbose):
                Write_log(log,' - %i round - train_metric: %.6f - val_metric: %.6f\n'%(i*self.verbose,evals_result_dic['training'][self.params['metric']][i*self.verbose],evals_result_dic['valid_1'][self.params['metric']][i*self.verbose]))
            all_valid_metric.append(Metric(trainDf.loc[val_index,label_name],val_pred))
            Write_log(log,'valid metric: %.8f\n'%all_valid_metric[-1])
            plt.plot(evals_result_dic['valid_1'][self.params['metric']],label='fold%s'%(fold))
        train_metric = Metric(train_target,train_pred)
        print('all train oof metric:%.8f'%(train_metric))
        valid_metric = Metric(trainDf[label_name],valid_df[label_name])
        print('all valid oof metric:%.8f'%(valid_metric))
        mean_valid_metric = np.mean(all_valid_metric)
        print('all valid mean metric:%.8f'%(mean_valid_metric))
        feature_importance_df = feature_importance_df.groupby(['feature_name']).mean().reset_index()
        feature_importance_df = feature_importance_df.sort_values(by=['importance_gain'],ascending=False)
        #feature_importance_df = pd.merge(feature_importance_df,colsDf,how='left',on='feature_name')
        feature_importance_df.to_csv('%s/valid/feature_importance_%slgb_metric_%.8f_%.8f_%.8f.csv'%(root,prefix,mean_valid_metric,valid_metric,train_metric),index=False)
        valid_df.to_csv('%s/valid/valid_%slgb_metric_%.8f_%.8f_%.8f.csv'%(root,prefix,mean_valid_metric,valid_metric,train_metric),index=False)
        plt.legend()
        plt.savefig('%s/log/valid_%slgb_metric_%.8f_%.8f_%.8f.png'%(root,prefix,mean_valid_metric,valid_metric,train_metric))
        plt.close('all')
        submission_df[label_name] = submission_df[[col for col in submission_df.columns if col != id_name]].mean(axis=1)
        submission_df[[id_name,label_name]].to_csv('%s/submission/submission_%slgb_metric_%.8f_%.8f_%.8f.csv'%(root,prefix,mean_valid_metric,valid_metric,train_metric),index=False)
        log.close()
        os.rename('%s/log/train.log'%root, '%s/log/%slgb_metric_%.8f_%.8f_%.8f.log'%(root,prefix,mean_valid_metric,valid_metric,train_metric))
        return None
