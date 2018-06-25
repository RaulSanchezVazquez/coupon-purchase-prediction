#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 11:55:18 2018

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

''' Train '''
scale_pos_weight = (y == 0).sum() / (y == 1).sum()
eval_set = [(X_val, y_val)]

xgb_model = xgb.XGBClassifier(
    learning_rate=.1,
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

model_path = os.path.join(config.SOURCE, 'models/xgb_model.pkl')
pickle.dump(xgb_model, open(model_path, 'wb'))

train_logs = xgb_model.evals_result()

''' Evaluation '''
test_y_pred = xgb_model.predict(X_test)
test_y_pred_proba = xgb_model.predict_proba(X_test)
class_report = eval_utils.class_report(
    y_true=y_test, 
    y_pred=test_y_pred,
    y_score=test_y_pred_proba)
print(class_report)


''' Feature Importance '''
shap_values = shap.TreeExplainer(xgb_model).shap_values(X)
shap.summary_plot(shap_values, X)