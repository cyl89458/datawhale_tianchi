# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 09:58:32 2020

@author: Yvonne Liu
"""
import matplotlib
import pandas as pd
import numpy as np
pd.set_option('display.max_columns',None)

# raw_train = pd.read_csv('./raw_data/used_car_train_20200313.csv', sep=' ')
# raw_test = pd.read_csv('./raw_data/used_car_testA_20200313.csv', sep=' ')
# raw_train['train'] = 1
# raw_test['train'] = 0
# all_raw_data = pd.concat([raw_train,raw_test],ignore_index=True)

"""
缺失異常處理
    1.剔除訓練集中 model 缺失的 1 筆數據
    2.BodyType, fuelType, gearbox 以眾數補全
        (以訓練集眾數補全，但依據訓練及測試集數據分布來看，兩者分布基本相同故眾數應相同)
    3.notRepairedDamage：'-'部分需替換為空值，並將整個字段轉為float形式；空值視為第三類別=2
    4.power：剔除>600部分；訓練集 power==0部分視為缺失，並用訓練集 power 以 bodyType 分組後各組中位數來補全
特徵構造
    1."age" = createDate年份 - regDate年份
    2."age_bin" = "age"每五年分一箱
    3.price在各brand的統計量
    4.price在各bodyType的統計量
"""

"""缺失異常處理"""

### 以各變量在訓練集中的眾數補全 (bodyType, fuelType, gearbox)
def fillna_with_mode_of_train(data, column):
    train_col_mode = data[data.train==1][column].mode()[0]
    filled_col = data[column].fillna(train_col_mode)
    return filled_col

### '-'部分需替換為空值，並將整個字段轉為float形式 (notRepairedDamage)
def null_str_to_null_float(data, column, null_str='-'):
    result_col = data[column].apply(lambda x : float(x) if x!=null_str else np.nan)
    return result_col

### 以指定變量做分組，並以訓練集各組目標缺失變量的中位數來補全缺失部分 (power)
def fillna_with_grouped_train_median(data, group_col, missing_col):
    missingCol_median_grouped_by_groupCol = data[data.train==1].groupby(group_col).agg({missing_col:'median'})
    filled_col = data.apply(lambda x : missingCol_median_grouped_by_groupCol.loc[x[group_col]][missing_col] if np.isnan(x[missing_col]) else x[missing_col], axis=1)
    return filled_col

"""特徵構造"""

### 車齡 age 計算
def cal_car_age(data):
    age = data.apply(lambda x : int(str(x.creatDate)[:4])-int(str(x.regDate)[:4]), axis=1)
    return age

### 分箱(每x一箱, 所有<min一箱, 所有>max一箱)
def sep_bins(data, column, bin_size, min_=None, max_=None):
    if min_==None:
        min_ = data[column].min()
    if max_==None:
        max_ = data[column].max()
    data[column+'_minus_min'] = data[column] - min_ +1
    bin_col = data[column+'_minus_min'].apply(lambda x : np.ceil((max_-min_)/bin_size) if x>(max_-min_+1) else np.ceil(x/bin_size) if x>0 else 0)
    
    del data[column+'_minus_min']
    return bin_col
        
### 統計量特徵(只用訓練集計算)
def cal_grouped_stats(data, group_col, stat_col, \
                      stats=['median','mean','size','std','max','min']):
    grouped_stats = data[data.train==1].groupby([group_col])[stat_col].agg(stats).reset_index()
    grouped_stats.columns = [group_col]+[group_col+'_gp_'+stat_col+'_'+s for s in stats]
    stats_merged = data.merge(grouped_stats, on=group_col, how='left')
    return stats_merged