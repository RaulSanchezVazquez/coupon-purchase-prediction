#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 11:55:18 2018

@author: raulsanchez
"""

import os
import xgboost as xgb
import pandas as pd
import shap
import pickle
from sklearn.metrics import classification_report

import config
import model_datasets
import common.eval_utils

(X_train,  y_train,  X_val,  y_val,  X_test,  y_test,
_, _, _) = model_datasets.get_train_val_test()

for c in X_train.select_dtypes('category').columns:
    X_train[c] = X_train[c].cat.codes
    X_val[c] = X_val[c].cat.codes
    X_test[c] = X_test[c].cat.codes

''' Train '''
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
eval_set = [(X_val, y_val)]

xgb_model = xgb.XGBClassifier(
    learning_rate=.3,
    scale_pos_weight=scale_pos_weight,
    n_jobs=8,
    n_estimators=1000)

xgb_model.fit(
    X_train,
    y_train, 
    eval_metric="auc", 
    eval_set=eval_set, 
    verbose=True,
    early_stopping_rounds=50)

train_logs = xgb_model.evals_result()
pd.Series(train_logs['validation_0']['auc']).plot()

''' Evaluation '''
test_y_pred = xgb_model.predict(X_test)
test_y_pred_proba = xgb_model.predict_proba(X_test)
class_report = common.eval_utils.class_report(
    y_true=y_test, 
    y_pred=test_y_pred,
    y_score=test_y_pred_proba)
print(class_report)

model_path = os.path.join(config.SOURCE, 'models/xgb_model.pkl')
pickle.dump(xgb_model, open(model_path, 'wb'))

''' Feature Importance '''
shap_values = shap.TreeExplainer(xgb_model).shap_values(X_train)
shap.summary_plot(shap_values, X_train)