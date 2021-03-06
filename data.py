import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
import datetime
import os,sys,re
from multiprocessing import Process,Manager
from tqdm import tqdm
from utils import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

conf = Conf()
root = conf.root
id_name = conf.id_name
label_name = conf.label_name
random_seed = conf.random_seed
card_cols = ['card%s'%(i+1) for i in range(6)]
addr_cols = ['addr1','addr2']
email_cols = ['P_emaildomain','R_emaildomain']
c_cols = ['C%s'%(i+1) for i in range(14)]
d_cols = ['D%s'%(i+1) for i in range(15)]
m_cols = ['M%s'%(i+1) for i in range(9)]
v_cols = ['V%s'%(i+1) for i in range(339)]
id_cols = ['id_%s'%str(i+1).zfill(2) for i in range(38)]
device_cols = ['DeviceType','DeviceInfo']
cat_cols = card_cols+['ProductCD','addr1','addr2']+email_cols+m_cols+['id_%s'%str(i+1).zfill(2) for i in range(11,38)]+device_cols
TransactionDT_interval = [86400,86400*7]
np.random.seed(random_seed)

def Get_train_test(nrows=None):
    train_df1 = pd.read_csv('%s/data/train_identity.csv'%root)
    train_df2 = pd.read_csv('%s/data/train_transaction.csv'%root,nrows=nrows,dtype={'TransactionAmt': str})
    test_df1 = pd.read_csv('%s/data/test_identity.csv'%root)
    test_df2 = pd.read_csv('%s/data/test_transaction.csv'%root,nrows=nrows,dtype={'TransactionAmt': str})
    train_df = train_df2.merge(train_df1,how='left',on=id_name)
    test_df = test_df2.merge(test_df1,how='left',on=id_name)
    test_df[label_name] = 0
    return train_df,test_df

def Get_nan_features(df):
    df['nanNum'] = df.isna().sum(axis=1)
    df['cardNanNum'] = df[card_cols].isna().sum(axis=1)
    df['emailNanNum'] = df[email_cols].isna().sum(axis=1)
    df['cNanNum'] = df[c_cols].isna().sum(axis=1)
    df['dNanNum'] = df[d_cols].isna().sum(axis=1)
    df['mNanNum'] = df[m_cols].isna().sum(axis=1)
    df['vNanNum'] = df[v_cols].isna().sum(axis=1)
    df['idNanNum'] = df[id_cols].isna().sum(axis=1)
    df['deviceNanNum'] = df[device_cols].isna().sum(axis=1)
    for col in df.columns:
        if col in cat_cols:
            df[col] = df[col].fillna('-999').astype(str)
            df[col] = df[col].apply(lambda x:x.lower())
    for col in card_cols+addr_cols:
        df[col] = df[col].apply(lambda x:x.replace('.0',''))
    return df

def Get_new_features(df):
    # TransactionAmt
    df['TransactionAmtDecimal'] = df['TransactionAmt'].apply(lambda x:int(x.split('.')[1]))
    df['TransactionAmtDecimal5'] = df['TransactionAmtDecimal'] % 5
    df['TransactionAmtDecimalLength'] = df['TransactionAmt'].apply(lambda x:len(x.split('.')[1]))
    df['TransactionAmt'] = df['TransactionAmt'].astype(float)
    df['TransactionAmt5'] = df['TransactionAmt'] % 5
    df['TransactionAmtDivD3'] = df['TransactionAmt'] / (df['D3']+1)

    # emaildomain
    df['isProtonMail'] = 0 + ((df['P_emaildomain'] == 'protonmail.com') | (df['R_emaildomain']  == 'protonmail.com'))
    emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other', 'optonline.net': 'other',
          'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo',
          'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 'aim.com': 'aol', 'hotmail.de': 'microsoft',
          'centurylink.net': 'centurylink', 'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other',
          'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 'protonmail.com': 'other',
          'hotmail.fr': 'microsoft', 'windstream.net': 'other', 'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo',
          'yahoo.de': 'yahoo', 'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other',
          'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft', 'verizon.net': 'yahoo',
          'msn.com': 'microsoft', 'q.com': 'centurylink', 'prodigy.net.mx': 'att', 'frontier.com': 'yahoo',
          'anonymous.com': 'other', 'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo',
          'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 'bellsouth.net': 'other',
          'embarqmail.com': 'centurylink', 'cableone.net': 'other', 'hotmail.es': 'microsoft', 'mac.com': 'apple',
          'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other',
          'cox.net': 'other', 'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}
    us_emails = ['gmail', 'net', 'edu']
    for c in ['P_emaildomain', 'R_emaildomain']:
        df[c + '_bin'] = df[c].map(emails)
        df[c + '_suffix'] = df[c].map(lambda x: str(x).split('.')[-1])
        df[c + '_suffix'] = df[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')

    # TransactionDT
    df['day'] = df['TransactionDT'] // (3600*24)
    #df['christmasMonth'] = 0
    #df.loc[(df['day']<=25)|((df['day']>=366)&(df['day']<=390)),'christmasMonth'] = 1
    df['hour'] = hours = np.floor(df['TransactionDT']/3600) % 24
    df['dayOfWeek'] = np.floor(df['TransactionDT']/(3600*24)) % 7
    df['dayOfMonth'] = df['TransactionDT'].apply(lambda x:(datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')+datetime.timedelta(seconds = x)).day)
    df['openCardDay'] = df['day'] - df['D1']
    df['firstTranWaitDays'] = df['D1'] - df['D2']
    for t in TransactionDT_interval:
        df['TransactionDT_%s'%t] = df['TransactionDT'] // t * t

    # Browser
    df['lastestBrowser'] = 0
    df.loc[df["id_31"]=="samsung browser 7.0",'lastestBrowser']=1
    df.loc[df["id_31"]=="opera 53.0",'lastestBrowser']=1
    df.loc[df["id_31"]=="mobile safari 10.0",'lastestBrowser']=1
    df.loc[df["id_31"]=="google search application 49.0",'lastestBrowser']=1
    df.loc[df["id_31"]=="firefox 60.0",'lastestBrowser']=1
    df.loc[df["id_31"]=="edge 17.0",'lastestBrowser']=1
    df.loc[df["id_31"]=="chrome 69.0",'lastestBrowser']=1
    df.loc[df["id_31"]=="chrome 67.0 for android",'lastestBrowser']=1
    df.loc[df["id_31"]=="chrome 63.0 for android",'lastestBrowser']=1
    df.loc[df["id_31"]=="chrome 63.0 for ios",'lastestBrowser']=1
    df.loc[df["id_31"]=="chrome 64.0",'lastestBrowser']=1
    df.loc[df["id_31"]=="chrome 64.0 for android",'lastestBrowser']=1
    df.loc[df["id_31"]=="chrome 64.0 for ios",'lastestBrowser']=1
    df.loc[df["id_31"]=="chrome 65.0",'lastestBrowser']=1
    df.loc[df["id_31"]=="chrome 65.0 for android",'lastestBrowser']=1
    df.loc[df["id_31"]=="chrome 65.0 for ios",'lastestBrowser']=1
    df.loc[df["id_31"]=="chrome 66.0",'lastestBrowser']=1
    df.loc[df["id_31"]=="chrome 66.0 for android",'lastestBrowser']=1
    df.loc[df["id_31"]=="chrome 66.0 for ios",'lastestBrowser']=1
    return df

def Get_id_features(df):
    df['notFountNum'] = (df[['id_12','id_15','id_16','id_27','id_28','id_29']]=='notfound').sum(axis=1)
    def Get_os(x):
        os_list = ['windows','ios','mac','linux','android','func']
        for o in os_list:
            if o in x:
                return o
        return x
    df['os'] = df['id_30'].apply(lambda x:Get_os(x))
    def Get_chrome(x):
        if re.match('chrome *.* for android',x):
            return 'chrome for android'
        os_list = ['opera','samsung','android','chrome','safari','mobile safari','firefox','edge','google','ie']
        for o in os_list:
            if o in x:
                return o
        return x
    df['chrome'] = df['id_31'].apply(lambda x:Get_chrome(x))
    df['w'] = df['id_33'].apply(lambda x:x.split('x')[0] if 'x' in x else x).astype(int)
    df['h'] = df['id_33'].apply(lambda x:x.split('x')[1] if 'x' in x else x).astype(int)
    df['w-h'] = df['w'] - df['h']
    df['area'] = df['w'] * df['h']
    df['ratio'] = df['w'] / (1.0+df['h'])
    return df

def Get_sum_features(df):
    df['TF'] = df[['id_35','id_36','id_37','id_38']].sum(axis=1)
    df['M1-M9'] = df[['M%s'%i for i in range(1,10)]].sum(axis=1)
    df['nan-12-3'] = df[['V279','V280','V284','V285','V286','V287','V290','V291','V292','V293','V294','V295','V297','V298','V299','V302','V303','V304','V305','V306','V307','V308','V309','V310','V311','V312','V316','V317','V318','V319','V320','V321']].sum(axis=1)
    df['nan-134-0'] = df[['V95','V96','V97','V98','V99','V100','V101','V102','V103','V104','V105','V106','V107','V108','V109','V110','V111','V112','V113','V114','V115','V116','V117','V118','V119','V120','V121','V122','V123','V124','V125','V126','V127','V128','V129','V130','V131','V132','V133','V134','V135','V136','V137',]].sum(axis=1)
    df['nan-1269-6031'] = df[['D1','V281','V282','V283','V288','V289','V296','V300','V301','V313','V314','V315']].sum(axis=1)
    df['nan-77096-12899'] = df[['V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','V29','V30','V31','V32','V33','V34','V53','V54','V55','V56','V57','V58','V59','V60','V61','V62','V63','V64','V65','V66','V67','V68','V69','V70','V71','V72','V73','V74']].sum(axis=1)
    df['nan-89164-12081'] = df[['V75','V76','V77','V78','V79','V80','V81','V82','V83','V84','V85','V86','V87','V88','V89','V90','V91','V92','V93','V94']].sum(axis=1)
    df['nan-168922-76851'] = df[['V35','V36','V37','V38','V39','V40','V41','V42','V43','V44','V45','V46','V47','V48','V49','V50','V51','V52']].sum(axis=1)
    df['nan-279287-176518'] = df[['D11','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','D2']].sum(axis=1)
    df['nan-271100-176639'] = df[['M1','M2','M3']].sum(axis=1)
    df['nan-346252-235004'] = df[['M8','M9']].sum(axis=1)
    df['nan-446307-364784'] = df['id_01'].astype(str) + df['id_12']
    df['nan-449121-369375'] = df[['V220','V221','V222','V227','V234','V238','V239','V245','V250','V251','V255','V256','V259','V270','V271','V272']].sum(axis=1)
    df['nan-449555-369714'] = df[['id_15','id_35','id_36','id_37','id_38']].sum(axis=1)
    df['nan-449562-369913'] = df['id_11'].astype(str) + df[['id_28','id_29']].sum(axis=1)
    df['nan-450721-370316'] = df[['V169','V170','V171','V174','V175','V180','V184','V185','V188','V189','V194','V195','V197','V198','V200','V201','V208','V209','V210']].sum(axis=1)
    df['nan-450909-369957'] = df[['V167','V168','V172','V173','V176','V177','V178','V179','V181','V182','V183','V186','V187','V190','V191','V192','V193','V196','V199','V202','V203','V204','V205','V206','V207','V211','V212','V213','V214','V215','V216']].sum(axis=1)
    df['nan-453675-371941'] = df[['id_05','id_06']].sum(axis=1)
    df['nan-460110-379963'] = df[['V217','V218','V219','V223','V224','V225','V226','V228','V229','V230','V231','V232','V233','V235','V236','V237','V240','V241','V242','V243','V244','V246','V247','V248','V249','V252','V253','V254','V257','V258','V260','V261','V262','V263','V264','V265','V266','V267','V268','V269','V273','V274','V275','V276','V277','V278']].sum(axis=1)
    df['nan-508189-430260'] = df[['V322','V323','V324','V325','V326','V327','V328','V329','V330','V331','V332','V333','V334','V335','V336','V337','V338','V339']].sum(axis=1)
    df['nan-508589-430636'] = df[['V143','V144','V145','V150','V151','V152','V159','V160','V164','V165','V166']].sum(axis=1)
    df['nan-508595-430906'] = df[['V138','V139','V140','V141','V142','V146','V147','V148','V149','V153','V154','V155','V156','V157','V158','V161','V162','V163']].sum(axis=1)
    df['nan-515614-432353'] = df[['D8','D9','id_09','id_10']].sum(axis=1)
    df['nan-524216-440210'] = df[['id_03','id_14']].sum(axis=1)
    df['nan-585371-501629'] = df[['id_22','id_23','id_27']].sum(axis=1)
    df['nan-585385-501632'] = df[['id_07','id_08']].sum(axis=1)
    return df

def Get_card_group_features(df,cardInfo,prefix,trainNrows=None,agg=True,norm=False,encoding='count'):
    norm_cols = ['V258','C14','V201','C1','C13','V257','V246','V317','C8']
    if norm:
        for col in norm_cols:
            df['%s_%sNorm'%(prefix,col)] = df[[col,prefix]].groupby([prefix])[col].transform(Norm)
    if len(cardInfo) > 1:
        df[prefix] = df[cardInfo].apply(lambda x:'_'.join([str(sub_x) for sub_x in x]),axis=1)
    if agg:
        for col in ['TransactionAmt','id_02']:
            if col in cardInfo:
                continue
            df['%s_%sDivCount'%(prefix,col)] = df[col].astype(float) / df[prefix].map(dict(df[prefix].value_counts()))
            df['%s_%sDivMean'%(prefix,col)] = df[col].astype(float) / df[[col,prefix]].groupby([prefix])[col].transform('mean')
            df['%s_%sDivMax'%(prefix,col)] = df[col].astype(float) / df[[col,prefix]].groupby([prefix])[col].transform('max')
        if train_k >= 0:
            tmp_tr = df[:train_nrows].groupby([prefix])[id_name].count()
            intersection_value = set(tmp_tr[tmp_tr>train_k].index) & set(df[train_nrows:][prefix].unique())
            df.loc[~df[prefix].isin(intersection_value),prefix] = np.nan
            print('%s no nan train count:%s, test count:%s'%(prefix,sum((~df[:train_nrows][prefix].isna())),sum((~df[train_nrows:][prefix].isna()))))
        for col in ['TransactionAmt','id_02','D3']:
            if col in cardInfo:
                continue
            df['%s_%sMean'%(prefix,col)] = df[[col,prefix]].groupby([prefix])[col].transform('mean')
            df['%s_%sStd'%(prefix,col)] = df[[col,prefix]].groupby([prefix])[col].transform('std')
            df['%s_%sSKew'%(prefix,col)] = df[[col,prefix]].groupby([prefix])[col].transform('skew')
            if col == 'D3':
                df['%s_D3Sum'%prefix] = df[['D3',prefix]].groupby([prefix])['D3'].transform('sum')
    if 'TransactionAmt' not in cardInfo:
        df['amt_%sNunique'%prefix] = df.groupby(['TransactionAmt','ProductCD'])[prefix].transform('nunique')
    else:
        df['%sDayNunique'%prefix] = df.groupby([prefix])['day'].transform('nunique')
    if len(cardInfo) > 1:
        if encoding == 'count':
            df = Count_encoding(df,[prefix])
        elif encoding == 'count_label':
            df = Count_label_encoding(df,[prefix])
        else:
            if 'count' in encoding:
                df = Count_encoding(df,[prefix],drop=False)
                encoding.remove('count')
            _,df = Mean_encoding(df,[prefix],encoding,drop=True)
    return df

def Get_diff_features(df,cardInfo,prefix,trainNrows):
    if len(cardInfo) > 1:
        df[prefix] = df[cardInfo].apply(lambda x:'_'.join([str(sub_x) for sub_x in x]),axis=1)
    if train_k >= 0:
        tmp_tr = df[:train_nrows].groupby([prefix])[id_name].count()
        intersection_value = set(tmp_tr[tmp_tr>train_k].index) & set(df[train_nrows:][prefix].unique())
        df.loc[~df[prefix].isin(intersection_value),prefix] = np.nan
        print('%s no nan train count:%s, test count:%s'%(prefix,sum((~df[:train_nrows][prefix].isna())),sum((~df[train_nrows:][prefix].isna()))))
    for col in ['TransactionDT','TransactionAmt','C5','C9','C13','V5','V99','V127','V186','V283','V294','V307','V310']:
        print('diff encoding %s by %s'%(prefix,col))
        df['%s_%sDiff'%(prefix,col)] = 0
        df.loc[:train_nrows,'%s_%sDiff'%(prefix,col)] = df.loc[:train_nrows][col] - df.loc[:train_nrows].groupby([prefix])[col].shift(1)
        df.loc[train_nrows:,'%s_%sDiff'%(prefix,col)] = df.loc[train_nrows:][col] - df.loc[train_nrows:].groupby([prefix])[col].shift(1)
        if col != 'V127':
            df['%s_%sDiffMin'%(prefix,col)] = df.groupby([prefix])['%s_%sDiff'%(prefix,col)].transform('min')
            df['%s_%sDiffMean'%(prefix,col)] = df.groupby([prefix])['%s_%sDiff'%(prefix,col)].transform('mean')
    df['%s_AmtDelV127Diff'%(prefix)] = df['TransactionAmt'] - df['%s_%sDiff'%(prefix,'V127')]
    df['%s_AmtDelV127DiffMean'%(prefix)] = df.groupby([prefix])['%s_AmtDelV127Diff'%(prefix)].transform('mean')
    df.drop(['%s_V127Diff'%(prefix),'%s_AmtDelV127Diff'%(prefix)],axis=1,inplace=True)
    if len(cardInfo) > 1:
        df.drop([prefix],axis=1,inplace=True)
    return df

for train_k in [-1,0,1]:
    train_df,test_df = Get_train_test(nrows=None)
    train_nrows = train_df.shape[0]
    train_df = Get_nan_features(train_df)
    test_df = Get_nan_features(test_df)
    tt_df = train_df.append(test_df).reset_index(drop=True)
    del train_df,test_df
    tt_df = Get_new_features(tt_df)
    tt_df = Get_card_group_features(tt_df,['TransactionAmt','ProductCD'],'amt',agg=False)
    tt_df = Get_card_group_features(tt_df,['dist2','TransactionAmt'],'dist2Amt',agg=False,encoding=['V258'])
    tt_df = Get_card_group_features(tt_df,['card1','TransactionAmt','ProductCD'],'card1Amt',agg=False)
    tt_df = Get_card_group_features(tt_df,['card3'],'card3',norm=True,agg=False,encoding=['C9','C13','D2','D11','D15','V10','V130','V257','V258'],trainNrows=train_nrows)
    tt_df = Get_card_group_features(tt_df,['card5'],'card5',norm=True,agg=False,encoding=['C9','C13','D2','D11','D15','V10','V130','V257','V258'],trainNrows=train_nrows)
    tt_df = Get_card_group_features(tt_df,['card1','TransactionAmt','ProductCD'],'card1Amt',agg=False,encoding=['count','V103'])
    tt_df = Get_card_group_features(tt_df,['card3','TransactionAmt','ProductCD'],'card3Amt',agg=False,encoding=['count','V103'])
    tt_df = Get_card_group_features(tt_df,['card3','card5','TransactionAmt','ProductCD'],'card35Amt',agg=False,encoding=['count','V103'])
    tt_df = Get_card_group_features(tt_df,['card1']+['openCardDay'],'card1OpenDay',encoding=['C9','C13','D2','D11','V10','V130','V257','V258'],trainNrows=train_nrows)
    tt_df = Get_card_group_features(tt_df,['card1']+['openCardDay']+['ProductCD'],'card1OpenDayProductCD',encoding=['C9','C13','D2','D11','V10','V130','V257','V258'],trainNrows=train_nrows)
    tt_df = Get_card_group_features(tt_df,['card1','card2','card3','card5']+['openCardDay'],'card1235OpenDay',encoding=['C9','C13','D2','D11','D15','V10','V130','V257','V258'],trainNrows=train_nrows)
    tt_df = Get_card_group_features(tt_df,card_cols+['openCardDay'],'uniqueCard0',encoding=['C9','C13','D2','D11','D15','V10','V130','V257','V258'],trainNrows=train_nrows)
    tt_df = Get_card_group_features(tt_df,card_cols+['openCardDay']+['ProductCD'],'uniqueCard0ProductCD',encoding=['C9','C13','D2','D11','D15','V10','V130','V257','V258'],trainNrows=train_nrows)
    tt_df = Get_card_group_features(tt_df,card_cols+['openCardDay']+['TransactionAmt','ProductCD'],'uniqueCard0Amt',agg=False)
    tt_df = Get_card_group_features(tt_df,card_cols+addr_cols+['openCardDay'],'uniqueCard1',encoding=['C9','C13','D2','D11','D15','V10','V130','V257','V258'],trainNrows=train_nrows)
    tt_df = Get_card_group_features(tt_df,card_cols+addr_cols+['openCardDay']+['ProductCD'],'uniqueCard1ProductCD')
    tt_df = Get_card_group_features(tt_df,card_cols+addr_cols+['openCardDay']+['TransactionAmt','ProductCD'],'uniqueCard1Amt',agg=False)
    tt_df = Get_card_group_features(tt_df,card_cols+addr_cols+email_cols+['openCardDay'],'uniqueCard2',encoding=['C9','C13','D2','D11','D15','V10','V130','V257','V258'],trainNrows=train_nrows)
    tt_df = Get_card_group_features(tt_df,card_cols+addr_cols+email_cols+['openCardDay']+['ProductCD'],'uniqueCard2ProductCD',encoding=['C9','C13','D2','D11','D15','V10','V130','V257','V258'],trainNrows=train_nrows)
    tt_df = Get_card_group_features(tt_df,card_cols+addr_cols+email_cols+['openCardDay']+['TransactionAmt','ProductCD'],'uniqueCard2Amt',agg=False)
    if train_k > -1:
        tt_df = Get_card_group_features(tt_df,['card1','P_emaildomain']+['openCardDay'],'card1PemailOpenDay',encoding=['C9','C13','D2','D11','D15','V10','V130','V257','V258'],trainNrows=train_nrows)
        tt_df = Get_card_group_features(tt_df,['card1','card2']+['openCardDay'],'card12OpenDay',encoding=['C5','C14','D1','D3','V103','V130','V257','V258'],trainNrows=train_nrows)
        tt_df = Get_card_group_features(tt_df,['card1','card2','card3']+['openCardDay'],'card123OpenDay',encoding=['C9','C13','D2','D11','D15','V10','V130','V257','V258'],trainNrows=train_nrows)
        tt_df = Get_card_group_features(tt_df,['card1','card2','card3','card5','P_emaildomain']+['openCardDay'],'card1235PemailOpenDay',encoding=['C9','C13','D2','D11','D15','V10','V130','V257','V258'],trainNrows=train_nrows)
    if train_k > 0:
        tt_df = Get_card_group_features(tt_df,email_cols+['openCardDay'],'emailOpenDay',encoding=['C9','C13','D2','D11','V10','V130','V257','V258'],trainNrows=train_nrows)
        tt_df = Get_card_group_features(tt_df,addr_cols+['openCardDay'],'addrOpenDay',encoding=['C9','C13','D2','D11','V10','V130','V257','V258'],trainNrows=train_nrows)
        tt_df = Get_card_group_features(tt_df,['card2']+['openCardDay'],'card2OpenDay',encoding=['C9','C13','D2','D11','V10','V130','V257','V258'],trainNrows=train_nrows)
        tt_df = Get_card_group_features(tt_df,['card1','card3'],'card13',encoding=['C9','C13','D2','D11','D15','V10','V130','V257','V258'],trainNrows=train_nrows)
        tt_df = Get_card_group_features(tt_df,['card3','card5'],'card35',encoding=['C9','C13','D2','D11','D15','V10','V130','V257','V258'],trainNrows=train_nrows)
    for t in TransactionDT_interval:
        tt_df = Get_card_group_features(tt_df,['TransactionAmt','ProductCD']+['TransactionDT_%s'%t],'interval%sAmt'%t,agg=False)
        tt_df = Get_card_group_features(tt_df,card_cols+addr_cols+email_cols+['openCardDay']+['TransactionDT_%s'%t],'interval%sUniqueCard2'%t,agg=False)
        tt_df = Get_card_group_features(tt_df,card_cols+addr_cols+email_cols+['openCardDay']+['TransactionAmt','ProductCD','TransactionDT_%s'%t],'interval%sUniqueCrad2Amt'%t,agg=False)
    tt_df = Get_id_features(tt_df)
    tt_df = Get_sum_features(tt_df)
    tt_df = Get_diff_features(tt_df,['card1']+['openCardDay'],'card1OpenDay',train_nrows)
    tt_df = Get_diff_features(tt_df,card_cols+addr_cols+email_cols+['openCardDay'],'uniqueCard2',train_nrows)
    if train_k > -1:
        tt_df = Get_diff_features(tt_df,['card1','card2']+['openCardDay'],'card12OpenDay',train_nrows)
        tt_df = Get_diff_features(tt_df,['card1','card2','card3','card5']+['openCardDay'],'card1235OpenDay',train_nrows)
        tt_df = Get_diff_features(tt_df,['card1','card2','card3','card5','P_emaildomain']+['openCardDay'],'card1235PemailOpenDay',train_nrows)
        tt_df = Get_diff_features(tt_df,card_cols+['openCardDay'],'uniqueCard0',train_nrows)
    if train_k > 0:
        tt_df = Get_diff_features(tt_df,card_cols+['ProductCD','openCardDay'],'uniqueCard0ProductCD',train_nrows)
        tt_df = Get_diff_features(tt_df,card_cols+addr_cols+['openCardDay'],'uniqueCard1',train_nrows)
        tt_df = Get_diff_features(tt_df,card_cols+addr_cols+['ProductCD','openCardDay'],'uniqueCard1ProductCD',train_nrows)
        tt_df = Get_diff_features(tt_df,card_cols+addr_cols+email_cols+['ProductCD','openCardDay'],'uniqueCard2ProductCD',train_nrows)
    for col in tt_df:
        if tt_df[col].dtype != 'object':
            tt_df[col] = tt_df[col].fillna(-999)
        else:
            tt_df[col ] = tt_df[col].fillna('-999')
    count_cols = ['day']
    count_label_cols = cat_cols + ['P_emaildomain_bin','P_emaildomain_suffix','R_emaildomain_bin','R_emaildomain_suffix','os','chrome','w','h','w-h','area','ratio','TF',\
            'M1-M9','nan-271100-176639','nan-346252-235004','nan-446307-364784','nan-449555-369714','nan-449562-369913','nan-585371-501629']
    mean_cols = card_cols + ['id_19','id_20','id_21','id_25','id_26']
    target_cols = []
    if not os.path.exists('%s/data/encodingTrain.csv'%(root)):
        mean_encoding_cols = []
        tt_df = Count_encoding(tt_df,count_cols,drop=True)
        new_cols,tt_df = Mean_encoding(tt_df,['dist2'],['TransactionAmt'],drop=False)
        mean_encoding_cols.extend(new_cols)
        new_cols,tt_df = Mean_encoding(tt_df,['card1','card2','card3','card5'],['TransactionAmt','V96','V103','V159','V178','V203','V265','V293','V317'],drop=False)
        mean_encoding_cols.extend(new_cols)
        new_cols,tt_df = Mean_encoding(tt_df,['id_19','id_20','id_21','id_25','id_26','id_33','DeviceInfo'],['id_02'],drop=True)
        mean_encoding_cols.extend(new_cols)
        count_label_cols = list(set(count_label_cols)-set(['card1','card2','card3','card5']+['id_19','id_20','id_21','id_25','id_26','id_33','DeviceInfo']))
        tt_df = Count_label_encoding(tt_df,count_label_cols)
        tt_df[:train_nrows][[id_name]+['%sCount'%col for col in count_cols]+count_label_cols+mean_encoding_cols].to_csv('%s/data/encodingTrain.csv'%(root),index=False)
        tt_df[train_nrows:][[id_name]+['%sCount'%col for col in count_cols]+count_label_cols+mean_encoding_cols].to_csv('%s/data/encodingTest.csv'%(root),index=False)
        tt_df.drop(['%sCount'%col for col in count_cols]+count_label_cols+mean_encoding_cols,axis=1,inplace=True)
    else:
        tt_df.drop(count_cols+count_label_cols+mean_cols+target_cols,axis=1,inplace=True)
    drop_cols = ['TransactionDT_%s'%t for t in TransactionDT_interval] + ['TransactionDT'] + ['openCardDay']
    tt_df.drop(drop_cols,axis=1,inplace=True)
    print(tt_df.head())
    tt_df[:train_nrows].to_csv('%s/data/new_train_k_gt_%s.csv'%(root,train_k),index=False)
    tt_df[train_nrows:].to_csv('%s/data/new_test_k_gt_%s.csv'%(root,train_k),index=False)
    del tt_df
