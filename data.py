import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
import datetime
import os,sys
from multiprocessing import Process,Manager
from collections import ChainMap
from utils import *

root = Conf().root

def Get_train_test():
    train_df1 = pd.read_csv('%s/data/train_identity.csv'%root)
    train_df2 = pd.read_csv('%s/data/train_transaction.csv'%root)
    test_df1 = pd.read_csv('%s/data/test_identity.csv'%root)
    test_df2 = pd.read_csv('%s/data/test_transaction.csv'%root)
    train_df = train_df2.merge(train_df1,how='left',on=id_name)
    test_df = test_df2.merge(test_df1,how='left',on=id_name)
    return train_df,test_df
