#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 22:30:12 2018

@author: raulsanchez
"""

def get_train_test():
    X, y = temporal_features.get_X_y()
    y = y['target']
                
    ''' Split train/val/test '''
    train_size = .8
    val_size = .1
    
    n = X.shape[0]
    
    idx_train = int(n * train_size)
    idx_val = int(idx_train * (1-val_size))
    
    X_train = X.iloc[:idx_val]
    y_train = y.iloc[:idx_val]
    
    X_val = X.iloc[idx_val:idx_train]
    y_val = y.iloc[idx_val:idx_train]
    
    X_test = X.iloc[idx_train:]
    y_test = y.iloc[idx_train:]
    
    print(X_train.shape[0]/X.shape[0], 
          X_train.reset_index()['date'].min(), 
          X_train.reset_index()['date'].max())
    print(X_val.shape[0]/X.shape[0], 
          X_val.reset_index()['date'].min(),
          X_val.reset_index()['date'].max())
    print(X_test.shape[0]/X.shape[0], 
          X_test.reset_index()['date'].min(),
          X_test.reset_index()['date'].max())