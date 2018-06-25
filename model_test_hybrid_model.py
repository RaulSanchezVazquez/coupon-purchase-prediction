#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 20:57:21 2018

@author: raulsanchez
"""

import os
import xgboost as xgb
import numpy as np
import pandas as pd
import shap
import pickle
from sklearn.metrics import classification_report

import config
import model_datasets
import common.eval_utils

(X_train,  y_train,  X_val,  y_val,  X_test,  y_test,
train, val, test) = model_datasets.get_train_val_test()

X_train = pd.concat([X_train, X_val], axis=0)
y_train = pd.concat([y_train, y_val], axis=0)

for c in X_train.select_dtypes('category').columns:
    X_train[c] = X_train[c].cat.codes
    X_test[c] = X_test[c].cat.codes

''' item features '''    
item_cols = []
for c in X_train.columns:
    if 'item_' in c[:5]:
        item_cols.append(c)
item_features = X_train.reset_index().sort_values(
    ['date', 'item_id']
).drop_duplicates(
    'item_id', 
    keep='last')[['item_id'] + item_cols]

''' user features '''
user_cols = []
for c in X_train.columns:
    if 'user_' in c[:5]:
        user_cols.append(c)
user_features = X_train.reset_index().sort_values(
    ['date', 'user_id']
).drop_duplicates(
    'user_id', 
    keep='last')[['user_id'] + user_cols]

user_features.set_index('user_id', inplace=True)
item_features.set_index('item_id', inplace=True)

''' Import models '''
xgb_model_path = os.path.join(config.SOURCE, 'models/xgb_model.pkl')
xgb_model = pickle.load(open(xgb_model_path, 'rb'))

recsys_model_path = os.path.join(config.SOURCE, 'models/recsys_model.pkl')
recsys_model = pickle.load(open(recsys_model_path, 'rb'))

candidate_users = X_test.reset_index()['user_id']
recsys_users = candidate_users

is_valid_user = recsys_users.isin(
    user_features.reset_index()['user_id'].values)
recsys_users = recsys_users[is_valid_user]

recsys_all_items = np.array(range(train.shape[1]))

X_test.reset_index(inplace=True)
X_test.set_index('user_id', inplace=True)

precision_at_10 = []
for user_it, user in enumerate(recsys_users.iloc[:2000]):

    recsys_predictions = recsys_model.predict(
        user_ids=[user], 
        item_ids=recsys_all_items, 
        num_threads=8)

    top_n = recsys_predictions.argsort()[:40]
    top_n_features = item_features.loc[top_n]
    
    local_user_features = user_features.loc[user]
    local_user_features = pd.DataFrame(
            local_user_features.values.reshape(1, -1).repeat(
                top_n_features.shape[0], axis=0),
            columns=local_user_features.index)
    
    user_item_features = pd.concat([
        local_user_features,
        top_n_features], 
        axis=1)[X_train.columns]
    
    y_prep_proba = xgb_model.predict_proba(user_item_features)
    
    recomendation = top_n[y_prep_proba[:, 1].argsort()[:10]]
    
    ground_truth = X_test.loc[[user]]
    
    precision = len(set(
            recomendation
        ).intersection(
            ground_truth.reset_index()['item_id']
        )
    )
    
    precision_at_10.append(precision)
    
    if (user_it % 100) == 0:
        print(pd.Series(precision_at_10).mean())