#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 18:21:10 2018

@author: raulsanchez
"""

import os
import pandas as pd
import pandas_profiling

import config
import datalayer

# 19413 x 24
items_catalog = datalayer.get('coupon_list_train')    
items_catalog_profile = pandas_profiling.ProfileReport(items_catalog)
items_catalog_profile.to_file(
    outputfile=os.path.join(
        config.SOURCE, 
        'docs/item_catalog.html')
)

# 2833180 x 8
log = datalayer.get('coupon_visit_train')
log_profile = pandas_profiling.ProfileReport(log)
log_profile.to_file(
    outputfile=os.path.join(
        config.SOURCE, 
        'docs/log_profile.html')
)
    
users = datalayer.get('user_list')
users_profile = pandas_profiling.ProfileReport(users)
users_profile.to_file(
    outputfile=os.path.join(
        config.SOURCE, 
        'docs/user_list.html')
)
