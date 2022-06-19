'''
This module is used to preprocess the raw data 
   
    1. Load the raw data
    2. Check and handle NaN values
    3. Fix the data frequency to be hourly
    4. Merge and normalize data
    5. Save data to processed data directory
'''


import pandas as pd
import numpy as np
import os
import math
import sys
import os
import random

from config import *

# set the method of replacing NaN values
FILL_NAN_WITH_ZERO = False
FILL_NAN_WITH_MEDIAN = False
FILL_NAN_WITH_MEAN = False
INTERPOLATE_NAN_VALUES = False
DELETE_NAN_ROWS = False
REPLACE_WITH_PREVIOUS_DAY = False
LINEAR_REGRESSION = True


def import_and_merge_data(input_filepath:str, output_filepath:str)-> None:
    ''' Imports data from raw data directory preprocesses data, extracts features and generates
    data for model training.
    
    :param intput_filepath: string path to raw data directory
    :param output_filepath: string path to output directory
    
    :Generated files:
    
        :merged.csv, merged_norm.csv: - Full dataset merged , normalized
    
        :pv.csv, pv_norm: - dataset with features required for PV model training

        :wp.csv, wp_norm: - dataset with features required for Wind model training / normalized
            

    :return: None 
    '''
    log("def import_and_merge_data")
    
    try:
        da_prices     = pd.read_csv(input_filepath + "da_prices.csv")
        loadcons      = pd.read_csv(input_filepath + "loadcons.csv")
        pv_power_gen  = pd.read_csv(input_filepath + "PV_power_gen.csv")
        radiation     = pd.read_csv(input_filepath + "radiation.csv")
        temperature   = pd.read_csv(input_filepath + "temperature.csv")
        windspeed     = pd.read_csv(input_filepath + "windspeed.csv")
        wind_power_gen= pd.read_csv(input_filepath + "wind_power_gen.csv")
        log("data imported successfully")
    except:
        log("Failed to import data")
        exit(1)
    
    
    pv_power_gen.rename(columns = {'Photovoltaic':'PV power'}, inplace = True)
    radiation.rename(columns = {'solar radiation - 5 min grouping - median':'Solar radiation'}, inplace = True)
    temperature.rename(columns = {'Temperature - 5 min grouping - median':'Temperature'}, inplace = True)
    windspeed.rename(columns = {'Wind speed - 5 min grouping - median':'Wind speed'}, inplace = True)
    wind_power_gen.rename(columns = {'Wind':'Wind power'}, inplace = True)
    
    df = temperature
    time = df["Time"][::12]
    time.reset_index(drop=True, inplace=True)

    df = df.merge(radiation,      on="Time", how="left")
    df = df.merge(windspeed,      on="Time", how="left")
    
    # Group by 1 hour and get the mean
    df = df.groupby(np.arange(len(df))//12).mean()

    df["Time"] = time
    # Merge with remaining data
    # By default in 1 hour group
    df = df.merge(pv_power_gen,   on="Time", how="left")
    df = df.merge(wind_power_gen, on="Time", how="left")
    df = df.merge(loadcons,       on="Time", how="left")
    df = df.merge(da_prices,      on="Time", how="left")

    df.index = pd.to_datetime(df['Time'], format='%d-%m-%Y %H:%M')
    for i in range(len(df.columns)):
        handle_nan_values(df, i)
    df.dropna(inplace = True)
    
    df['Seconds'] = df.index.map(pd.Timestamp.timestamp)
    day = 60*60*24
    
    df['Day sin']  = np.sin(df['Seconds'] * (2 * np.pi / day))
    df['Day cos']  = np.cos(df['Seconds'] * (2 * np.pi / day))
    
    df = df.drop('Seconds', axis=1)
    df = df.drop('Time', axis=1)
    
    df_pv = df[["Temperature","solar radiation","PV power","Day sin","Day cos"]].copy()
    df_wp = df[["Temperature","Wind speed","Wind power","Day sin","Day cos"]].copy()
    
    try:
        df.to_csv(output_filepath + "merged.csv", index=False)
        log('merged.csv saved in ' + output_filepath)
    except:
        log('Unable to save merged.csv file')
        
    for i in range (0,len(df.columns)):
        df = normalize_column(df, i)
    
    try:
        df.to_csv(output_filepath + "merged_and_normalized.csv", index=False)
        log('merged_and_normalized.csv saved in ' + output_filepath)
        
    except:
        log('Unable to save merged_and_normalized.csv file')
        
    try:
        df_pv.to_csv(output_filepath + "pv.csv", index=False)
        log('pv.csv saved in ' + output_filepath)
        
    except:
        log('Unable to save pv.csv file')
        
    for i in range (0,len(df_pv.columns)):
        df_pv = normalize_column(df_pv, i)
        
    try:
        df_pv.to_csv(output_filepath + "pv_norm.csv", index=False)
        log('pv_norm.csv saved in ' + output_filepath)
    except:
        log('Unable to save pv_norm.csv file')
        
    try:
        df_wp.to_csv(output_filepath + "wp.csv", index=False)
        log('wp.csv saved in ' + output_filepath)
    except:
        log('Unable to save wp.csv file')
        
    for i in range (0,len(df_wp.columns)):
        df_wp = normalize_column(df_wp, i)
        
    try:
        df_wp.to_csv(output_filepath + "wp_norm.csv", index=False)
        log('wp_norm.csv saved in ' + output_filepath)
    except:
        log('Unable to save wp_norm.csv file')
    
    return None   

def check_nan_values(df:pd.DataFrame, col:int=1):
    col_name = df.columns[col]
        
    for i in df[col_name]:
        is_NaN = math.isnan(i)
        if is_NaN:
            log("df contains NaN values")
            return True
    log("df does not contain NaN values")
    return False

def handle_nan_values(df: pd.DataFrame, col:int = 1):
    col_name = df.columns[col]
        
    if FILL_NAN_WITH_ZERO:
        df[col_name] = df[col_name].fillna(0)
    
    elif FILL_NAN_WITH_MEDIAN:
        df[col_name].fillna((df[col_name].median()), inplace=True)
    
    elif FILL_NAN_WITH_MEAN:
        df[col_name].fillna((df[col_name].mean()), inplace=True)
    
    elif INTERPOLATE_NAN_VALUES:
        df.interpolate(method ='linear', limit_direction ='forward')
    
    elif DELETE_NAN_ROWS:
        df.dropna(inplace = True)
    
    elif REPLACE_WITH_PREVIOUS_DAY:
        j=0  
        for i, row in df.iterrows():
            value = row[col_name]
            a = random.randint(1, 30) #pick random day in the past month
            if pd.isnull(value):
                df.at[i, col_name] = df.iloc[j-a*24][col_name]*random.uniform(0.95, 1.05)  #replace and add 10% noise
                j+=1

def normalize_column(df:pd.DataFrame, col:int=1, a:int=0, b:int=1):
    col_name = df.columns[col]
    df[col_name] = (df[col_name] - df[col_name].min())/(df[col_name].max() - df[col_name].min())
    df[col_name] = (b-a)*df[col_name]+a 
    return df 

if __name__ == '__main__':
    import_and_merge_data(raw_data_dir, processed_data_dir)
     
    
    















