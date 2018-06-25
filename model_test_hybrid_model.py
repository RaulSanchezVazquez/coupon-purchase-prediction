#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 20:57:21 2018

@author: raulsanchez
"""

import os
import xgboost as xgb
import shap
import pickle
from sklearn.metrics import classification_report

import config
import temporal_features
import eval_utils

X, y = temporal_features.get_X_y()
(u_encoder, 
 i_encoder, 
 train, 
 test) = temporal_features.get_train_test_matrix()

y = y['target']

percent_null = X.isnull().sum() / X.shape[0]
X.drop(percent_null[percent_null == 1].index, axis=1, inplace=True)

for c in X.select_dtypes('object').columns:
    X[c] = X[c].astype('category').cat.codes

''' Split train/val/test '''
train_size = .8
val_size = .1

n = X.shape[0]

idx_train = int(n * train_size)

X_train = X.iloc[:idx_train]
y_train = y.iloc[:idx_train]

X_test = X.iloc[idx_train:]
y_test = y.iloc[idx_train:]

X_train.reset_index(inplace=True)
X_test.reset_index(inplace=True)

item_cols = []
for c in X_train.columns:
    if 'item_' in c[:5]:
        item_cols.append(c)
user_cols = []
for c in X_train.columns:
    if 'user_' in c[:5]:
        user_cols.append(c)

item_features = X_train.sort_values(['date', 'item_id']).drop_duplicates(
    'item_id', 
    keep='last')[item_cols]

user_features = X_train.sort_values(['date', 'user_id']).drop_duplicates(
    'user_id', 
    keep='last')[user_cols]



user_features.set_index('user_id', inplace=True)
item_features.set_index('item_id', inplace=True)

''' Import models '''
xgb_model_path = os.path.join(config.SOURCE, 'models/xgb_model.pkl')
recsys_model_path = os.path.join(config.SOURCE, 'models/recsys_model.pkl')

xgb_model = pickle.load(open(xgb_model_path, 'rb'))
recsys_model = pickle.load(open(recsys_model_path, 'rb'))

candidate_users = X_test.reset_index()['user_id']
recsys_users = u_encoder.transform(candidate_users[:10])
recsys_all_items = i_encoder.transform(i_encoder.classes_)

X_test.reset_index(inplace=True)
X_test.set_index('user_id', inplace=True)

for user in recsys_users:break

    recsys_predictions = recsys_model.predict(
        user_ids=[user], 
        item_ids=recsys_all_items, 
        num_threads=1)

    top_n = recsys_predictions.argsort()[:1000]
    top_n_features = item_features.loc[i_encoder.inverse_transform(top_n)]
    
    top_n_features[top_n_features.columns[23]]
    
    X_train[]
    top_n = i_encoder.inverse_transform(top_n)
    
    ground_truth = X_test.loc[u_encoder.inverse_transform(user)]
    gt_item_ids = ground_truth['item_id'].values
    
    set(gt_item_ids).intersection(top_n)
    