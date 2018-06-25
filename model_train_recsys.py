#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 20:12:50 2018

@author: raulsanchez
"""

import os
import pickle
import pandas as pd
import numpy as np

from scipy.sparse import coo_matrix

from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score

import config
import model_datasets

(_,  _,  _,  _,  _,  _,
train, val, test) = model_datasets.get_train_val_test()

alpha = 1e-4
epochs = 120
model_path = os.path.join(config.SOURCE, 'models/recsys_model.pkl')
if not os.path.exists(model_path):
    recsys_model = LightFM(
        no_components=300,
        learning_rate=0.1,
        loss='warp',
        learning_schedule='adagrad',
        user_alpha=alpha,
        item_alpha=alpha
    )
    
    precision_hist = []
    for epoch in range(epochs):
        recsys_model.fit_partial(train, epochs=1, num_threads=8)
        precision = precision_at_k(recsys_model, val, num_threads=8).mean()
        print(f'[{epoch}] {precision}')
        precision_hist.append(precision)
    
    auc = auc_score(recsys_model, test, num_threads=8).mean()
    pd.Series(precision_hist).plot(
        grid=True, title=f'{auc} epochs: {len(precision_hist)}')
    
    model_path = os.path.join(config.SOURCE, 'models/recsys_model.pkl')
    pickle.dump(recsys_model, open(model_path, 'wb'))
else:
    recsys_model = pickle.load(open(model_path,'rb'))
    precision = precision_at_k(recsys_model, test, num_threads=8).mean()
    auc = auc_score(recsys_model, test, num_threads=8).mean()
    