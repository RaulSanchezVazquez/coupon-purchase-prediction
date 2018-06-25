#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 22:30:12 2018

@author: raulsanchez
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import coo_matrix

import config
import featurelayer

def get_train_val_test():
    """
    """
    
    data_path = os.path.join(
        config.SOURCE, 
        'data/train_val_test.hdf')
    
    if not os.path.exists(data_path):
        X, y = featurelayer.get_X_y()
        y = y['target']
        
        for c in X.select_dtypes('object').columns:
            X[c] = X[c].astype('category')
        
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
    
        print('Train')
        print(X_train.shape[0]/X.shape[0], 
              X_train.reset_index()['date'].min(), 
              X_train.reset_index()['date'].max())
        print('Val')
        print(X_val.shape[0]/X.shape[0], 
              X_val.reset_index()['date'].min(),
              X_val.reset_index()['date'].max())
        print('Test')
        print(X_test.shape[0]/X.shape[0], 
              X_test.reset_index()['date'].min(),
              X_test.reset_index()['date'].max())
        
        X_train.to_hdf(data_path, key='X_train', format='table')
        y_train.to_hdf(data_path, key='y_train', format='table')
        
        X_val.to_hdf(data_path, key='X_val', format='table')
        y_val.to_hdf(data_path, key='y_val', format='table')
        
        X_test.to_hdf(data_path, key='X_test', format='table')
        y_test.to_hdf(data_path, key='y_test', format='table')
    else:
        X_train  = pd.read_hdf(data_path, key='X_train')
        y_train = pd.read_hdf(data_path, key='y_train')
        
        X_val = pd.read_hdf(data_path, key='X_val')
        y_val = pd.read_hdf(data_path, key='y_val')
        
        X_test = pd.read_hdf(data_path, key='X_test')
        y_test = pd.read_hdf(data_path, key='y_test')
    
    last_user = max([
        X_train.reset_index()['user_id'].max(),
        X_val.reset_index()['user_id'].max(),
        X_test.reset_index()['user_id'].max()])
    
    last_item = max([
        X_train.reset_index()['item_id'].max(),
        X_val.reset_index()['item_id'].max(),
        X_test.reset_index()['item_id'].max()])
        
    ''' Make recsys matrices '''
    # Train
    data = np.concatenate([
        np.ones(X_train.shape[0]), [0] ])
    row = np.concatenate([
        X_train.reset_index()['user_id'].values, [last_user]])
    col = np.concatenate([
        X_train.reset_index()['item_id'].values, [last_item]])
    train = coo_matrix((data, (row, col)))
    
    # Validation
    data = np.concatenate([
        np.ones(X_val.shape[0]), [0] ])
    row = np.concatenate([
        X_val.reset_index()['user_id'].values, [last_user]])
    col = np.concatenate([
        X_val.reset_index()['item_id'].values, [last_item]])
    val = coo_matrix((data, (row, col)))
    
    # Test
    data = np.concatenate([
        np.ones(X_test.shape[0]), [0] ])
    row = np.concatenate([
        X_test.reset_index()['user_id'].values, [last_user]])
    col = np.concatenate([
        X_test.reset_index()['item_id'].values, [last_item]])
    test = coo_matrix((data, (row, col)))
    
    return X_train, y_train, X_val, y_val, X_test, y_test, train, val, test