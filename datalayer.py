#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 10:57:00 2018

@author: raulsanchez
"""

import os
import json
import pandas as pd

import config

ITEM_ID = 'COUPON_ID_hash'
USER_ID = 'USER_ID_hash'
TARGET_F = 'PURCHASE_FLG'

def get(dataset_name):
    """
    Data Fetcher.
    It already takes care of returning datetime as date objects and
    handles traductions from japanese to english when traduction is
    available.
    
    Parameters
    ----------
    
    dataset_name : str
        Dataset name 
    
    Return
    ------
    data : pandas.DataFrame
        Dataset
    """
    translate_path = os.path.join(
        config.SOURCE,
        'data/challenge_ds/translations.json')
    
    translate_cols = []
    date_cols = []
    translate = json.load(open(translate_path))
    
    if dataset_name == 'coupon_area_train':
        data_path = os.path.join(
            config.SOURCE,
            'data/challenge_ds/coupon_area_train.csv')
        data = pd.read_csv(data_path)
        
        translate_cols = ['SMALL_AREA_NAME', 'PREF_NAME']
        
    if dataset_name == 'coupon_list_train':
        data_path = os.path.join(
            config.SOURCE,
            'data/challenge_ds/coupon_list_train.csv')
        data = pd.read_csv(data_path)
        translate_cols = [
            'large_area_name', 'ken_name', 'small_area_name',
            'CAPSULE_TEXT', 'GENRE_NAME']
        date_cols = [
            'DISPFROM', 'DISPEND',
            'VALIDEND', 'VALIDFROM']
        
    if dataset_name == 'coupon_visit_train':
        data_path = os.path.join(
            config.SOURCE,
            'data/challenge_ds/coupon_visit_train.csv')
        data = pd.read_csv(data_path)
        date_cols = ['I_DATE']
        
    if dataset_name == 'user_list':
        data_path = os.path.join(
            config.SOURCE,
            'data/challenge_ds/user_list.csv')
        data = pd.read_csv(data_path)
        translate_cols = ['PREF_NAME']
        date_cols = ['REG_DATE', 'WITHDRAW_DATE']

    for c in translate_cols:
        data[c] = data[c].apply(
            lambda x: translate[x] if x in translate else x)
    
    for c in date_cols:
        data[c] = pd.to_datetime(data[c])
        
    return data
    
def get_USER_Features():
    """
    Compute user's features.
    
    Return
    ------
    users : pandas.DataFrame
        User features
    """
    users = get('user_list')

    return users

def get_ITEM_Features():
    """
    Compute user's features.
    
    Return
    ------
    users : pandas.DataFrame
        User features
    """
    
    items = get('coupon_list_train')
    
    items['DISCOUNT_PRICE_percentage'] = (
        items['DISCOUNT_PRICE'] / items['CATALOG_PRICE'])

    return items

def get_user_item_meta(train_size=.8):
    """
    """
    data_path = os.path.join(
        config.SOURCE, 
        'data/user_item_meta.hdf')
    
    if not os.path.exists(data_path):
        user_item = get('coupon_visit_train')
        item_features = get_ITEM_Features()
        user_features = get_USER_Features()
        
        item_features.set_index(ITEM_ID, inplace=True)
        user_features.set_index(USER_ID, inplace=True)
        
        user_features = user_features.loc[user_item[USER_ID]]
        user_features.index = user_item.index
        
        item_features = item_features.loc[user_item['VIEW_' + ITEM_ID]]
        item_features.index = user_item.index
        
        ui_meta = pd.concat([
            user_item,
            item_features,
            user_features], axis=1)
        
        ui_meta.to_hdf(data_path, key='ui_meta')
    else:
        ui_meta = pd.read_hdf(data_path, key='ui_meta')
    
    return ui_meta


#def get_user_item_matrix():
#    """
#    """
#    global ITEM_ID
#    
#    user_item_path = os.path.join(
#        config.SOURCE, 'data/user_item.hdf')
#    
#    user_enc_path = os.path.join(
#            config.SOURCE, 'data/user_encoder.pkl')
#    
#    item_enc_path = os.path.join(
#            config.SOURCE, 'data/item_encoder.pkl')
#    
#    ITEM_ID = "VIEW_" + ITEM_ID
#    
#    if not os.path.exists(user_item_path):
#        user_item = get('coupon_visit_train')[[
#            'I_DATE', 
#            ITEM_ID, 
#            USER_ID]]
#        
#        user_item.sort_values('I_DATE', inplace=True)
#        
#        # Encode users
#        user_encoder = LabelEncoder().fit(
#            user_item[USER_ID])
#        
#        user_item[USER_ID] = user_encoder.transform(
#            user_item[USER_ID])
#        
#        # Encode Items
#        item_encoder = LabelEncoder().fit(
#            user_item[ITEM_ID])
#        
#        user_item[ITEM_ID] = item_encoder.transform(
#            user_item[ITEM_ID])
#        
#        # Cache data
#        pickle.dump(user_encoder, open(user_enc_path, 'wb'))
#        pickle.dump(item_encoder, open(item_enc_path, 'wb'))
#        user_item.to_hdf(user_item_path, key='user_item')
#    else:
#        user_encoder = pickle.load(open(user_enc_path, 'rb'))
#        item_encoder = pickle.load(open(item_enc_path, 'rb'))
#        user_item = pd.read_hdf(user_item_path, key='user_item')
#    
#    return user_encoder, item_encoder, user_item
#
#def get_train_test_user_item_matrix(train_size=.8):
#    """
#    """
#    global ITEM_ID
#    
#    ITEM_ID = ITEM_ID
#    
#    _, _, user_item = get_user_item_matrix()
#    
#    last_user = user_item[USER_ID].max()
#    last_item = user_item[ITEM_ID].max()
#    
#    # Filter users with minimum hist. sizes
#    min_hist_size = 1
#    users_hist_size = user_item[USER_ID].value_counts()
#    candidate_users = users_hist_size[
#        (users_hist_size > min_hist_size)
#    ].index
#    filter_users =  user_item[USER_ID].isin(candidate_users)
#    user_item = user_item[filter_users]
#    
#    # Split train/test
#    n_transactions = user_item.shape[0]
#    
#    train_boundary = int(n_transactions * train_size)
#    train_user_item = user_item.iloc[:train_boundary]
#    test_user_item = user_item.iloc[train_boundary:]
#    
#    # Create train UserxItem matrix
#    data = np.concatenate([
#        np.ones(train_user_item.shape[0]), [0] ])
#    row = np.concatenate([
#        train_user_item[USER_ID].values, [last_user]])
#    col = np.concatenate([
#        train_user_item[ITEM_ID].values, [last_item]])
#    
#    train_user_item = coo_matrix((data, (row, col)))
#    
#    # Create test UserxItem  matrix
#    data = np.concatenate([
#        np.ones(test_user_item.shape[0]), [0] ])
#    row = np.concatenate([
#        test_user_item[USER_ID].values, [last_user]])
#    col = np.concatenate([
#        test_user_item[ITEM_ID].values, [last_item]])
#    
#    test_user_item = coo_matrix((data, (row, col)))
#    
#    return train_user_item, test_user_item
