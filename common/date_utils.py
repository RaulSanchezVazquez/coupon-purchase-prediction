#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 18:59:44 2018

@author: raulsanchez
"""
import pandas as pd
import numpy as np

def date_expand(feature, time=False, prefix=None, sep='__'):
    """
    Converts a pd.Series column of type datetime-like (datetime64) to many 
    columns containing the various features with summary information.
    Parameters:
    -----------
    feature : pandas.Series
        Date column from witch to expand to various features.
        If it is not a datetime64 series, it will be converted to one with 
        pd.to_datetime.
        
    time : Bool
        If True, the time features: Hour, Minute, Second will be added.
    
    prefix : string
        if None the default column name is used as prefix for feature names.
    sep : string
        Separator between 'prefix' and 'feature_name'
    
    Returns:
    --------
    new_features : pandas.DataFrame
        The expanded version of the timeseries as dataframe
    
    Examples:
    ---------
    import pandas as pd
    import numpy as np
    import re
    
    df = pd.DataFrame({ 'A' : pd.to_datetime([
        '3/11/2000', '3/12/2000', '3/13/2000'], 
        infer_datetime_format=False) })
    
    print(df['A'])
    
    out:
            A
        0   2000-03-11
        1   2000-03-12
        2   2000-03-13
    print(date_expand(df['A']))
    out:
           A__Day  A__Dayofweek  A__Dayofyear  A__Is_month_end  A__Is_month_start  \
        0      11             5            71            False              False   
        1      12             6            72            False              False   
        2      13             0            73            False              False   
        
           A__Is_quarter_end  A__Is_quarter_start  A__Is_year_end  A__Is_year_start  \
        0              False                False           False             False   
        1              False                False           False             False   
        2              False                False           False             False   
        
           A__Month  A__Week  A__Year  A__Elapsed  
        0         3       10     2000   952732800  
        1         3       10     2000   952819200  
        2         3       11     2000   952905600  
        
    """
    
    #List of new features to create
    attr = [
        'Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
        'Is_month_end', 'Is_month_start', 
        'Is_quarter_end', 'Is_quarter_start', 
        'Is_year_end', 'Is_year_start']
    
    if time: 
        #If time is available
        attr = attr + ['Hour', 'Minute', 'Second']
        
    feature = feature.copy()
    
    is_datetime = np.issubdtype(
        feature.dtype,
        np.datetime64)
    
    if not is_datetime:
        feature = pd.to_datetime(
            feature, 
            infer_datetime_format=True)
    
    new_features = {}
    for n in attr:
        new_features[n] = getattr(feature.dt, n.lower())
    
    new_features = pd.DataFrame(new_features)
    new_features['Elapsed'] = feature.astype(np.int64) // 10 ** 9
    
    if prefix is None:
        prefix = feature.name
    
    new_features = new_features.add_prefix("%s%s" % (prefix, sep))
    
    return new_features