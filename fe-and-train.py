# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# Any results you write to the current directory are saved as output.

# General imports
import numpy as np
import pandas as pd
import os, sys, gc, warnings, random

from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm

import math
warnings.filterwarnings('ignore')

## -------------------
## Seeder
# :seed to make all processes deterministic     # type: int
def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
## ------------------- 

########################### Vars ################################
SEED = 42
seed_everything(SEED)
LOCAL_TEST = False
TARGET = 'isFraud'

########################### DATA LOAD ########################################
print('Load Data')
train_df = pd.read_pickle('data/processed-cis/train_transaction.pkl')

if LOCAL_TEST:
    test_df = train_df.iloc[-100000:,].reset_index(drop=True)
    train_df = train_df.iloc[:400000,].reset_index(drop=True)
    
    train_identity = pd.read_pickle('data/processed-cis/train_identity.pkl')
    test_identity  = train_identity[train_identity['TransactionID'].isin(test_df['TransactionID'])].reset_index(drop=True)
    train_identity = train_identity[train_identity['TransactionID'].isin(train_df['TransactionID'])].reset_index(drop=True)
else:
    test_df = pd.read_pickle('data/processed-cis/test_transaction.pkl')
    train_identity = pd.read_pickle('data/processed-cis/train_identity.pkl')
    test_identity = pd.read_pickle('data/processed-cis/test_identity.pkl')
    
base_columns = list(train_df) + list(train_identity)

########################### 'P_emaildomain' - 'R_emaildomain'
# Lets do small check to see if 'P_emaildomain' - 'R_emaildomain' matters
# and if NaN matters
if LOCAL_TEST:
    print('#'*5, 'Email matchings test')
    for df in [train_df, test_df]:
        len_match = df[df['P_emaildomain']==df['R_emaildomain']]
        len_not_match = df[df['P_emaildomain']!=df['R_emaildomain']]

        print('Match', len(len_match[len_match[TARGET]==1])/len(len_match), 'Total items:', len(len_match))
        print('Not Match', len(len_not_match[len_not_match[TARGET]==1])/len(len_not_match), 'Total items:', len(len_not_match))
        print('#'*5)

    print('#'*5, 'Email NaN test')
    for df in [train_df, test_df]:
        len_match = df[(df['P_emaildomain'].isna())&(df['R_emaildomain'].isna())]
        len_not_match = df[['P_emaildomain','R_emaildomain',TARGET]].dropna()

        print('All NaN', len(len_match[len_match[TARGET]==1])/len(len_match), 'Total items:', len(len_match))
        print('No NaNs', len(len_not_match[len_not_match[TARGET]==1])/len(len_not_match), 'Total items:', len(len_not_match))
        print('#'*5)  
        
########################### 'P_emaildomain' - 'R_emaildomain'
# Matching
train_df['email_check'] = np.where(train_df['P_emaildomain']==train_df['R_emaildomain'],1,0)
test_df['email_check']  = np.where(test_df['P_emaildomain']==test_df['R_emaildomain'],1,0)

# All NaNs
train_df['email_check_nan_all'] = np.where((train_df['P_emaildomain'].isna())&(train_df['R_emaildomain'].isna()),1,0)
test_df['email_check_nan_all']  = np.where((test_df['P_emaildomain'].isna())&(test_df['R_emaildomain'].isna()),1,0)

# Any NaN
train_df['email_check_nan_any'] = np.where((train_df['P_emaildomain'].isna())|(train_df['R_emaildomain'].isna()),1,0)
test_df['email_check_nan_any']  = np.where((test_df['P_emaildomain'].isna())|(test_df['R_emaildomain'].isna()),1,0)

# Fix NaN, get "prefix"
def fix_emails(df):
    df['P_emaildomain'] = df['P_emaildomain'].fillna('email_not_provided')
    df['R_emaildomain'] = df['R_emaildomain'].fillna('email_not_provided')
    
    df['email_match_not_nan'] = np.where((df['P_emaildomain']==df['R_emaildomain'])&
                                     (df['P_emaildomain']!='email_not_provided'),1,0)
    
    df['P_email_prefix'] = df['P_emaildomain'].apply(lambda x: x.split('.')[0])
    df['R_email_prefix'] = df['R_emaildomain'].apply(lambda x: x.split('.')[0])
    return df

train_df = fix_emails(train_df)
test_df = fix_emails(test_df)

########################### D9 and TransactionDT
# Seems that D9 column is a hour
# But what hour?
# Local time? Server time? Shop time?
# Previous transaction? Most common time for client?
# Is there difference between TransactionDT and D9 column?
# Is there connection with distance?
train_df['local_hour'] = train_df['D9']*24
test_df['local_hour']  = test_df['D9']*24

train_df['local_hour'] = train_df['local_hour'] - (train_df['TransactionDT']/(60*60))%24
test_df['local_hour']  = test_df['local_hour'] - (test_df['TransactionDT']/(60*60))%24

train_df['local_hour_dist'] = train_df['local_hour']/train_df['dist2']
test_df['local_hour_dist']  = test_df['local_hour']/test_df['dist2']

########################### M columns (except M4)
# All these columns are binary encoded 1/0
# We can have some features from it
i_cols = ['M1','M2','M3','M5','M6','M7','M8','M9']

train_df['M_sum'] = train_df[i_cols].sum(axis=1).astype(np.int8)
test_df['M_sum']  = test_df[i_cols].sum(axis=1).astype(np.int8)

train_df['M_na'] = train_df[i_cols].isna().sum(axis=1).astype(np.int8)
test_df['M_na']  = test_df[i_cols].isna().sum(axis=1).astype(np.int8)

train_df['M_type'] = ''
test_df['M_type']  = ''

for col in i_cols:
    train_df['M_type'] = '_'+train_df[col].astype(str)
    test_df['M_type'] = '_'+test_df[col].astype(str)
    
########################### C columns
# C columns are some counts, based on client identity
# Most popular Value is "1" -> that seems to be just a single match 
# (New or stable client)
# You can check that auc score for that cliens are lower than global
# Lets encode such client types

i_cols = ['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14']

train_df['C_sum'] = 0
test_df['C_sum']  = 0

train_df['C_null'] = 0
test_df['C_null']  = 0

for col in i_cols:
    train_df['C_sum'] += np.where(train_df[col]==1,1,0)
    test_df['C_sum']  += np.where(test_df[col]==1,1,0)

    train_df['C_null'] += np.where(train_df[col]==0,1,0)
    test_df['C_null']  += np.where(test_df[col]==0,1,0)
    
    valid_values = train_df[col].value_counts()
    valid_values = valid_values[valid_values>1000]
    valid_values = list(valid_values.index)
    
    train_df[col+'_valid'] = np.where(train_df[col].isin(valid_values),1,0)
    test_df[col+'_valid']  = np.where(test_df[col].isin(valid_values),1,0)
    
########################### Reset values for "noise" card1
valid_card = train_df['card1'].value_counts()
valid_card = valid_card[valid_card>10]
valid_card = list(valid_card.index)
    
train_df['card1'] = np.where(train_df['card1'].isin(valid_card), train_df['card1'], np.nan)
test_df['card1']  = np.where(test_df['card1'].isin(valid_card), test_df['card1'], np.nan)

########################### Device info
train_identity['DeviceInfo'] = train_identity['DeviceInfo'].fillna('unknown_device').str.lower()
test_identity['DeviceInfo'] = test_identity['DeviceInfo'].fillna('unknown_device').str.lower()

train_identity['DeviceInfo_c'] = train_identity['DeviceInfo']
test_identity['DeviceInfo_c'] = test_identity['DeviceInfo']

device_match_dict = {
    'sm':'sm-',
    'sm':'samsung',
    'huawei':'huawei',
    'moto':'moto',
    'rv':'rv:',
    'trident':'trident',
    'lg':'lg-',
    'htc':'htc',
    'blade':'blade',
    'windows':'windows',
    'lenovo':'lenovo',
    'linux':'linux',
    'f3':'f3',
    'f5':'f5'
}
for dev_type_s, dev_type_o in device_match_dict.items():
    train_identity['DeviceInfo_c'] = train_identity['DeviceInfo_c'].apply(lambda x: dev_type_s if dev_type_o in x else x)
    test_identity['DeviceInfo_c'] = test_identity['DeviceInfo_c'].apply(lambda x: dev_type_s if dev_type_o in x else x)

train_identity['DeviceInfo_c'] = train_identity['DeviceInfo_c'].apply(lambda x: 'other_d_type' if x not in device_match_dict else x)
test_identity['DeviceInfo_c'] = test_identity['DeviceInfo_c'].apply(lambda x: 'other_d_type' if x not in device_match_dict else x)

########################### Device info 2
train_identity['id_30'] = train_identity['id_30'].fillna('unknown_device').str.lower()
test_identity['id_30'] = test_identity['id_30'].fillna('unknown_device').str.lower()

train_identity['id_30_c'] = train_identity['id_30']
test_identity['id_30_c'] = test_identity['id_30']

device_match_dict = {
    'ios':'ios',
    'windows':'windows',
    'mac':'mac',
    'android':'android'
}
for dev_type_s, dev_type_o in device_match_dict.items():
    train_identity['id_30_c'] = train_identity['id_30_c'].apply(lambda x: dev_type_s if dev_type_o in x else x)
    test_identity['id_30_c'] = test_identity['id_30_c'].apply(lambda x: dev_type_s if dev_type_o in x else x)
    
train_identity['id_30_v'] = train_identity['id_30'].apply(lambda x: ''.join([i for i in x if i.isdigit()]))
test_identity['id_30_v'] = test_identity['id_30'].apply(lambda x: ''.join([i for i in x if i.isdigit()]))
        
train_identity['id_30_v'] = np.where(train_identity['id_30_v']!='', train_identity['id_30_v'], 0).astype(int)
test_identity['id_30_v'] = np.where(test_identity['id_30_v']!='', test_identity['id_30_v'], 0).astype(int)

########################### Browser
train_identity['id_31'] = train_identity['id_31'].fillna('unknown_br').str.lower()
test_identity['id_31']  = test_identity['id_31'].fillna('unknown_br').str.lower()

train_identity['id_31'] = train_identity['id_31'].apply(lambda x: x.replace('webview','webvw'))
test_identity['id_31']  = test_identity['id_31'].apply(lambda x: x.replace('webview','webvw'))

train_identity['id_31'] = train_identity['id_31'].apply(lambda x: x.replace('for',' '))
test_identity['id_31']  = test_identity['id_31'].apply(lambda x: x.replace('for',' '))

browser_list = set(list(train_identity['id_31'].unique()) + list(test_identity['id_31'].unique()))
browser_list2 = []
for item in browser_list:
    browser_list2 += item.split(' ')
browser_list2 = list(set(browser_list2))

browser_list3 = []
for item in browser_list2:
    browser_list3 += item.split('/')
browser_list3 = list(set(browser_list3))

for item in browser_list3:
    train_identity['id_31_e_'+item] = np.where(train_identity['id_31'].str.contains(item),1,0).astype(np.int8)
    test_identity['id_31_e_'+item] = np.where(test_identity['id_31'].str.contains(item),1,0).astype(np.int8)
    if train_identity['id_31_e_'+item].sum()<100:
        del train_identity['id_31_e_'+item], test_identity['id_31_e_'+item]
        
train_identity['id_31_v'] = train_identity['id_31'].apply(lambda x: ''.join([i for i in x if i.isdigit()]))
test_identity['id_31_v'] = test_identity['id_31'].apply(lambda x: ''.join([i for i in x if i.isdigit()]))
        
train_identity['id_31_v'] = np.where(train_identity['id_31_v']!='', train_identity['id_31_v'], 0).astype(int)
test_identity['id_31_v'] = np.where(test_identity['id_31_v']!='', test_identity['id_31_v'], 0).astype(int)

########################### Merge Identity columns
temp_df = train_df[['TransactionID']]
temp_df = temp_df.merge(train_identity, on=['TransactionID'], how='left')
del temp_df['TransactionID']
train_df = pd.concat([train_df,temp_df], axis=1)
    
temp_df = test_df[['TransactionID']]
temp_df = temp_df.merge(test_identity, on=['TransactionID'], how='left')
del temp_df['TransactionID']
test_df = pd.concat([test_df,temp_df], axis=1)

########################### Freq encoding
i_cols = ['card1','card2','card3','card5',
          'C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14',
          'D1','D2','D3','D4','D5','D6','D7','D8','D9',
          'addr1','addr2',
          'dist1','dist2',
          'P_emaildomain', 'R_emaildomain',
          'id_01','id_02','id_03','id_04','id_05','id_06','id_07','id_08','id_09','id_10',
          'id_11','id_13','id_14','id_17','id_18','id_19','id_20','id_21','id_22','id_24',
          'id_25','id_26','id_30','id_31','id_32','id_33','id_33_0','id_33_1',
          'DeviceInfo','DeviceInfo_c','id_30_c','id_30_v','id_31_v',
         ]

for col in i_cols:
    temp_df = pd.concat([train_df[[col]], test_df[[col]]])
    fq_encode = temp_df[col].value_counts().to_dict()   
    train_df[col+'_fq_enc'] = train_df[col].map(fq_encode)
    test_df[col+'_fq_enc']  = test_df[col].map(fq_encode)
    
########################### ProductCD and M4 Target mean
for col in ['ProductCD','M4']:
    temp_dict = train_df.groupby([col])[TARGET].agg(['mean']).reset_index().rename(
                                                        columns={'mean': col+'_target_mean'})
    temp_dict.index = temp_dict[col].values
    temp_dict = temp_dict[col+'_target_mean'].to_dict()

    train_df[col+'_target_mean'] = train_df[col].map(temp_dict)
    test_df[col+'_target_mean']  = test_df[col].map(temp_dict)

########################### Encode Str columns
for col in list(train_df):
    if train_df[col].dtype=='O':
        print(col)
        train_df[col] = train_df[col].fillna('unseen_before_label')
        test_df[col]  = test_df[col].fillna('unseen_before_label')
        
        train_df[col] = train_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)
        
        le = LabelEncoder()
        le.fit(list(train_df[col])+list(test_df[col]))
        train_df[col] = le.transform(train_df[col])
        test_df[col]  = le.transform(test_df[col])
        
        train_df[col] = train_df[col].astype('category')
        test_df[col] = test_df[col].astype('category')

########################### TransactionAmt

# Let's add some kind of client uID based on cardID ad addr columns
# The value will be very specific for each client so we need to remove it
# from final feature. But we can use it for aggregations.
train_df['uid'] = train_df['card1'].astype(str)+'_'+train_df['card2'].astype(str)+'_'+train_df['card3'].astype(str)+'_'+train_df['card4'].astype(str)
test_df['uid'] = test_df['card1'].astype(str)+'_'+test_df['card2'].astype(str)+'_'+test_df['card3'].astype(str)+'_'+test_df['card4'].astype(str)

train_df['uid2'] = train_df['uid'].astype(str)+'_'+train_df['addr1'].astype(str)+'_'+train_df['addr2'].astype(str)
test_df['uid2'] = test_df['uid'].astype(str)+'_'+test_df['addr1'].astype(str)+'_'+test_df['addr2'].astype(str)

# Check if Transaction Amount is common or not (we can use freq encoding here)
# In our dialog with model we are telling to trust or not to these values  
valid_card = train_df['TransactionAmt'].value_counts()
valid_card = valid_card[valid_card>10]
valid_card = list(valid_card.index)
    
train_df['TransactionAmt_check'] = np.where(train_df['TransactionAmt'].isin(test_df['TransactionAmt']), 1, 0)
test_df['TransactionAmt_check']  = np.where(test_df['TransactionAmt'].isin(train_df['TransactionAmt']), 1, 0)

# For our model current TransactionAmt is a noise (even when features importances are telling contrariwise)
# There are many unique values and model doesn't generalize well
# Lets do some aggregations
i_cols = ['card1','card2','card3','card5','uid','uid2']

for col in i_cols:
    for agg_type in ['mean', 'std']:
        new_col_name = col+'_TransactionAmt_'+agg_type
        temp_df = pd.concat([train_df[[col, 'TransactionAmt']], test_df[[col,'TransactionAmt']]])
        temp_df = temp_df.groupby([col])['TransactionAmt'].agg([agg_type]).reset_index().rename(
                                                columns={agg_type: new_col_name})
        
        temp_df.index = list(temp_df[col])
        temp_df = temp_df[new_col_name].to_dict()   
    
        train_df[new_col_name] = train_df[col].map(temp_df)
        test_df[new_col_name]  = test_df[col].map(temp_df)

# Small "hack" to transform distribution 
# (doesn't affect auc much, but I like it more)
# please see how distribution transformation can boost your score 
# (not our case but related)
# https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html
train_df['TransactionAmt'] = np.log1p(train_df['TransactionAmt'])
test_df['TransactionAmt'] = np.log1p(test_df['TransactionAmt'])

########################### Anomaly Search in geo information

# Let's look on bank addres and client addres matching
# card3/card5 bank country and name?
# Addr2 -> Clients geo position (country)
# Most common entries -> normal transactions
# Less common etries -> some anonaly
train_df['bank_type'] = train_df['card3'].astype(str)+'_'+train_df['card5'].astype(str)
test_df['bank_type']  = test_df['card3'].astype(str)+'_'+test_df['card5'].astype(str)

train_df['address_match'] = train_df['bank_type'].astype(str)+'_'+train_df['addr2'].astype(str)
test_df['address_match']  = test_df['bank_type'].astype(str)+'_'+test_df['addr2'].astype(str)

for col in ['address_match','bank_type']:
    temp_df = pd.concat([train_df[[col]], test_df[[col]]])
    temp_df[col] = np.where(temp_df[col].str.contains('nan'), np.nan, temp_df[col])
    temp_df = temp_df.dropna()
    fq_encode = temp_df[col].value_counts().to_dict()   
    train_df[col] = train_df[col].map(fq_encode)
    test_df[col]  = test_df[col].map(fq_encode)

train_df['address_match'] = train_df['address_match']/train_df['bank_type'] 
test_df['address_match']  = test_df['address_match']/test_df['bank_type']

########################### Features elimination 
from scipy.stats import ks_2samp
features_check = []
columns_to_check = set(list(train_df)).difference(base_columns)
for i in columns_to_check:
    features_check.append(ks_2samp(test_df[i], train_df[i])[1])

features_check = pd.Series(features_check, index=columns_to_check).sort_values() 
features_discard = list(features_check[features_check==0].index)
print(features_discard)

########################### Model Features 
## We can use set().difference() but order matters
## Matters only for deterministic results
## In case of remove() we will not change order
## even when variable will be renamed
## please see this link to see how set is ordered
## https://stackoverflow.com/questions/12165200/order-of-unordered-python-sets
rm_cols = [
    'TransactionID','TransactionDT', # These columns are pure noise right now
    TARGET,                          # Not target in features))
    'uid','uid2',                    # Our new clien uID -> very noisy data
    'bank_type',                     # Victims bank could differ by time
]
features_columns = list(train_df)
for col in rm_cols + features_discard:
    if col in features_columns:
        features_columns.remove(col)
        
########################### Model params
lgb_params = {
                    'objective':'binary',
                    'boosting_type':'gbdt',
                    'metric':'auc',
                    'n_jobs':-1,
                    'learning_rate':0.01,
                    'num_leaves': 2**8,
                    'max_depth':-1,
                    'tree_learner':'serial',
                    'colsample_bytree': 0.7,
                    'subsample_freq':1,
                    'subsample':1,
                    'n_estimators':800,
                    'max_bin':255,
                    'verbose':-1,
                    'seed': SEED,
                    'early_stopping_rounds':100, 
                } 

########################### Model
import lightgbm as lgb

def make_predictions(tr_df, tt_df, features_columns, target, lgb_params, NFOLDS=2):
    folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)

    X,y = tr_df[features_columns], tr_df[target]    
    P,P_y = tt_df[features_columns], tt_df[target]  

    tt_df = tt_df[['TransactionID',target]]    
    predictions = np.zeros(len(tt_df))
    
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
        print('Fold:',fold_)
        tr_x, tr_y = X.iloc[trn_idx,:], y[trn_idx]
        vl_x, vl_y = X.iloc[val_idx,:], y[val_idx]
            
        print(len(tr_x),len(vl_x))
        tr_data = lgb.Dataset(tr_x, label=tr_y)

        if LOCAL_TEST:
            vl_data = lgb.Dataset(P, label=P_y) 
        else:
            vl_data = lgb.Dataset(vl_x, label=vl_y)  

        estimator = lgb.train(
            lgb_params,
            tr_data,
            valid_sets = [tr_data, vl_data],
            verbose_eval = 200,
        )   
        
        pp_p = estimator.predict(P)
        predictions += pp_p/NFOLDS

        if LOCAL_TEST:
            feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importance(),X.columns)), columns=['Value','Feature'])
            print(feature_imp)
        
        del tr_x, tr_y, vl_x, vl_y, tr_data, vl_data
        gc.collect()
        
    tt_df['prediction'] = predictions
    
    return tt_df
## -------------------

########################### Model Train
if LOCAL_TEST:
    test_predictions = make_predictions(train_df, test_df, features_columns, TARGET, lgb_params)
    print(metrics.roc_auc_score(test_predictions[TARGET], test_predictions['prediction']))
else:
    lgb_params['learning_rate'] = 0.005
    lgb_params['n_estimators'] = 1800
    lgb_params['early_stopping_rounds'] = 100    
    test_predictions = make_predictions(train_df, test_df, features_columns, TARGET, lgb_params, NFOLDS=10)
    
########################### Export
if not LOCAL_TEST:
    test_predictions['isFraud'] = test_predictions['prediction']
    test_predictions[['TransactionID','isFraud']].to_csv('submission.csv', index=False)