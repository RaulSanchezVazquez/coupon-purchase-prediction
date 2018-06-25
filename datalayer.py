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
