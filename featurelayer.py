#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 02:06:15 2018

@author: raulsanchez
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import config
import datalayer
from common import parallel

ITEM_ID = 'VIEW_COUPON_ID_hash'
USER_ID = 'USER_ID_hash'
TARGET_F = 'PURCHASE_FLG'

users_grp = None
items_grp = None

temp_features_folder = os.path.join(
    config.SOURCE, 'data/temporal_features/')

if not os.path.exists(temp_features_folder):
    os.mkdir(temp_features_folder)

def process_temp_features():
    """
    """
    global users_grp
    global items_grp
    
    users_already_computed = [
        file.split('/')[-1].replace('.csv', '')
        for file in glob.glob(temp_features_folder + "*")]
    
    # Read data
    ui_meta = datalayer.get_user_item_meta()
    
    # Consider users with more than one item on hist
    user_hist_size = ui_meta[USER_ID].value_counts()
    candidate_users = user_hist_size[user_hist_size > 1].index
    
    ui_meta = ui_meta[ui_meta[USER_ID].isin(candidate_users)]
    
    users_grp = ui_meta.groupby(USER_ID)
    items_grp = ui_meta.groupby(ITEM_ID)
    
    missing_users_features = list(set(
        ui_meta[USER_ID].unique().tolist()
        ) - set(
        users_already_computed))
    
    parallel.apply(
        get_temporal_features,
        missing_users_features,
        n_jobs=8)
    
def get_temporal_features(user_id):
    '''
    user_id = '2a06587e4c97810c282f0474c6e1f7ea'
    '''
    try:
        global users_grps
        global items_grp
        
        user_hist = users_grp.get_group(user_id)
        user_hist.sort_values('I_DATE', inplace=True)
        
        features = []
        
        for idx, hist_item in user_hist.iloc[1:].iterrows():
            ref_date = hist_item['I_DATE']
            
            item_id = hist_item[ITEM_ID]
            
            ''' Item temporal features '''
            # Item views
            item_hist = items_grp.get_group(item_id)
            item_hist = item_hist[item_hist['I_DATE'] < ref_date]
            
            item_n_views = item_hist.shape[0]
            
            # Price
            item_percentage = hist_item['DISCOUNT_PRICE_percentage']
            item_price = hist_item['CATALOG_PRICE']
            
            # Item age
            age_norm = 60 * 60 # hour scale
            item_age_promo = (
                ref_date - hist_item['DISPFROM']
                ).total_seconds() / age_norm
            
            # Item location
            item_area = hist_item['large_area_name']
            item_ken = hist_item['ken_name']
            item_small_area = hist_item['small_area_name']
            
            # Item purchase population stats
            item_purchases = item_hist[item_hist[TARGET_F] == 1]
            
            # Sex
            try:
                purchase_sex_f, purchases_sex_m = (
                    item_purchases['SEX_ID'].value_counts().loc[['f', 'm']]
                ).fillna(0)
            except:
                purchase_sex_f, purchases_sex_m = 0, 0
            
            # Age
            item_buyers_age_mean = item_purchases['AGE'].mean()
            item_buyers_age_median = item_purchases['AGE'].median()
            item_buyers_age_std = item_purchases['AGE'].std()
            
            # Area
            item_area_purchases = {}
            for f in ['large_area_name', 'small_area_name', 'ken_name']:
                try:
                    area_purchase_cnt = item_purchases[f].value_counts()
                    most_buys = area_purchase_cnt.iloc[0]
                    most_buys2 = np.nan
                
                    if area_purchase_cnt.shape[0] > 1:
                        most_buys2 = area_purchase_cnt.iloc[1]
                except:
                    most_buys = np.nan
                    most_buys2 = np.nan
                
                item_area_purchases['item_%s_most_buy' % f] = most_buys
                item_area_purchases['item_%s_most_buy2' % f] = most_buys2
            
            ''' User temporal features '''
            user_hist_local = user_hist[user_hist['I_DATE'] < ref_date]
            
            is_same_user_item = (user_hist_local[ITEM_ID] == item_id)
            user_same_item_n_views = is_same_user_item.sum()
            user_same_item_n_purchases = user_hist_local[
                is_same_user_item
            ][TARGET_F].sum()
            
            user_purchases_same_item_area = (
                user_hist_local['large_area_name'] == item_area
            ).sum()
            
            user_purchases_same_item_ken_name = (
                user_hist_local['ken_name'] == item_ken).sum()
            
            user_purchases_same_item_small_area = (
                user_hist_local['small_area_name'] == item_small_area
            ).sum()
            
            user_last_item = user_hist_local.iloc[-1]
            
            features_local = {
            'date': ref_date, 
            'user_id': user_id, 
            'item_id': item_id,
            
            'item_age_promo': item_age_promo,
            
            'item_n_views': item_n_views,
            'item_n_purchases': item_purchases.shape[0],
            
            'item_area': item_area,
            'item_ken': item_ken,
            'item_small_area': item_small_area,
            
            'item_ken_most_buy': item_area_purchases['item_ken_name_most_buy'],
            'item_ken_most_buy2': item_area_purchases['item_ken_name_most_buy2'],
            'item_large_area_most_buy': item_area_purchases['item_large_area_name_most_buy'],
            'item_large_area_most_buy2': item_area_purchases['item_large_area_name_most_buy2'],
            'item_small_area_most_buy': item_area_purchases['item_small_area_name_most_buy'],
            'item_small_area_most_buy2': item_area_purchases['item_small_area_name_most_buy2'],
            
            'item_buyers_age_mean': item_buyers_age_mean,
            'item_buyers_age_median': item_buyers_age_median,
            'item_buyers_age_std': item_buyers_age_std,
            
            'item_DISCOUNT_PRICE_percentage': item_percentage,
            'item_price': item_price,
            
            'item_purchase_sex_f': purchase_sex_f, 
            'item_purchases_sex_m': purchases_sex_m,
            
            'user_same_item_n_views': user_same_item_n_views,
            'user_same_item_n_purchases': user_same_item_n_purchases,
            
            'user_last_item_large_area_name': user_last_item['large_area_name'],
            'user_last_item_ken_name': user_last_item['ken_name'],
            'user_last_item_small_area_name': user_last_item['small_area_name'],
            
            'user_purchases_same_item_area': user_purchases_same_item_area,
            'user_purchases_same_item_ken_name': user_purchases_same_item_ken_name,
            'user_purchases_same_item_small_area': user_purchases_same_item_small_area,
            
            'user_AGE': user_hist['AGE'].iloc[0],
            
            'user_SEX_ID': user_hist['SEX_ID'].iloc[0],
            'target': hist_item[TARGET_F]
            }
            
            features.append(features_local)
        
        features = pd.DataFrame(features)
        
        features.to_csv(
            os.path.join(temp_features_folder, user_id+".csv")
        )
    except:
        print(user_id)
    
def read_csv_helper(x):
    return pd.read_csv(x)

def get_X_y():
    data_path = os.path.join(
        config.SOURCE, 
        'data/X_y_temporal.hdf')
    
    if not os.path.exists(data_path):
        data = parallel.apply(
            read_csv_helper,
            glob.glob(temp_features_folder + "*"),
            n_jobs=8)
        
        data = pd.concat(data, axis=0)
        data.sort_values('date', inplace=True)
        
        u_encoder = LabelEncoder().fit(data['user_id'])
        i_encoder = LabelEncoder().fit(data['item_id'])
        
        data['user_id'] = u_encoder.transform(data['user_id'])
        data['item_id'] = i_encoder.transform(data['item_id'])
    
        data.set_index(['user_id', 'item_id' ,'date'], inplace=True)
        
        percent_null = data.isnull().sum() / data.shape[0]
        data.drop(
            percent_null[percent_null == 1].index, 
            axis=1, 
            inplace=True)
    
        y = data[['target']]
        X = data.drop(['target', 'Unnamed: 0'], axis=1)
        
        y.to_hdf(data_path, key='y')
        X.to_hdf(data_path, key='X')
        
        u_enc_path = os.path.join(config.SOURCE, 'data/u_encoder.pkl')
        i_enc_path = os.path.join(config.SOURCE, 'data/i_encoder.pkl')
        
        pickle.dump(u_encoder, open(u_enc_path, 'wb'))
        pickle.dump(i_encoder, open(i_enc_path, 'wb'))
    else:
        y = pd.read_hdf(data_path, key='y')
        X = pd.read_hdf(data_path, key='X')
    
        X.drop([
            'user_same_item_n_views',
            'user_same_item_n_purchases',
            'user_purchases_same_item_area',
            'user_purchases_same_item_ken_name',
            'user_purchases_same_item_small_area'],
            inplace=True, axis=1)
    
    return X, y