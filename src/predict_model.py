##
# @package predict_model - Use best model to predict PV power for the next 48 hours
# 
#

import numpy as np
import pandas as pd
import sys
from config import *
from keras.models import model_from_json

## Load specified model by name from models_dir
# @param model_name:str - Name of the model
# return model:keras.models
def load_model_from_json(model_name:str="model"):   # load model from json file
    json_file = open(model_dir + model_name + '.json', 'r') # open json file
    loaded_model_json = json_file.read()             # read json file
    json_file.close()                           # close json file
    log(f"Loaded model from {model_dir + model_name + '.json'}")    # log model name
    # load json and create model
    model = model_from_json(loaded_model_json)  # load json file
    # load weights into model
    model.load_weights(model_dir + model_name + ".h5")
    log(f"Loaded weights from {model_dir + model_name + '.h5'}")
    return model

##  Normalize dataframe column on range[a,b]
#   @param df:pd.DataFrame - Dataframe to normalize
#   @param col:int - Index to column
#   @param a:int - Min value (default 0)
#   @param b:int - Max value (default 1) 
#   @return df:pd.DataFrame - Normalized dataframe
def normalize_column(df:pd.DataFrame, col:int = 1, a:int=0, b:int=1): 
    col_name = df.columns[col]  # get the column name
    df[col_name] = (df[col_name] - df[col_name].min())/(df[col_name].max() - df[col_name].min())  # normalize
    df[col_name] = (b-a)*df[col_name]+a  # scale to [a,b]
    return df

##  Reverse normalize dataframe column from past range[a,b]
#   @param df:pd.DataFrame - Dataframe to normalize
#   @param col:int - Index to column
#   @param a:int - Min value (default 0)
#   @param b:int - Max value (default 1)
#   @return df:pd.DataFrame - Normalized dataframe
def reverse_normalize(df:pd.DataFrame, col_name:str, a:int=0, b:int=1)->pd.DataFrame:
    merged = pd.read_csv(processed_data_dir + "merged.csv")  # read the dataframe
    x_min = merged[col_name].min() # get the min value
    x_max = merged[col_name].max() # get the max value
    df[col_name] = (df[col_name]*(x_max-x_min) + x_min)/(b-a) - a # reverse normalize
    return df

##  Add sine and cosine to seconds of day and normalize dataframe
#   @param df:pd.DataFrame - Dataframe to normalize
#   @return df:pd.DataFrame - processed dataframe
def add_day_sin_cos_and_normalize(df:pd.DataFrame)->pd.DataFrame:
  df.index = pd.to_datetime(df['Time'], format='%d-%m-%Y %H:%M')  # convert the index to datetime
  df['Seconds'] = df.index.map(pd.Timestamp.timestamp)  # convert the index to seconds
  day = 60*60*24  # get the number of seconds in a day
  df['Day sin']  = np.sin(df['Seconds'] * (2 * np.pi / day))  # add the sin of the seconds
  df['Day cos']  = np.cos(df['Seconds'] * (2 * np.pi / day))  # add the cos of the seconds
  df = df.drop('Seconds', axis=1) # remove the seconds column
  df = df.drop("Time", axis=1)  # remove the time column
  for i in range (0,len(df.columns)): # iterate over the columns
      df = normalize_column(df, i)  # normalize the column
  return df   # return the normalized dataframe

##  Add sine and cosine to seconds of day and normalize dataframe
#   @param df:pd.DataFrame - Dataframe to normalize
#   @param seconds:int - keep datapoints after this timestamp
#   @return df:pd.DataFrame - processed dataframe
def keep_next_24_hours_data(df:pd.DataFrame, seconds:int)->pd.DataFrame:
    df.index = pd.to_datetime(df['Time'], format='%d-%m-%Y %H:%M')
    df['Seconds'] = df.index.map(pd.Timestamp.timestamp)
    df= df.loc[df["Seconds"] >= seconds]
    df = df.sort_values(by='Seconds')
    df = df.drop('Seconds', axis=1)
    return df

##  Predict next hour of the target feature
#   @param df:pd.DataFrame - Dataframe to normalize
#   @param model:keras.models - model to be used for the the prediction
#   @param look_back:int - timesteps to look back
#   @param target:string - feature to predict
#   @return df:pd.DataFrame - processed dataframe
def predict_next_hour(df:pd.DataFrame, model, look_back:int=24, target:str="PV power")->np.array:
  y_index = df.columns.get_loc(target) # get the index of the y column
  df_as_np = df.to_numpy()  # convert the dataframe to a numpy array
  X = []  # create the X array
  y = []  # create the y array
  i = len(df_as_np)-look_back # iterate over the data
  row = [r for r in df_as_np[i:i+look_back]]  # get the row
  X.append(row) # add the row to the X array
  label = df_as_np[i+look_back-1][y_index]  # get the label
  y.append(label) # add the label to the y array
  X = np.array(X) # convert the X array to a numpy array
  y = np.array(y) # convert the y array to a numpy array
  pred = model.predict(X).flatten() # make the prediction
  return pred # return the prediction

##  Function to make PV power predictions for a period of the available forecast weather data \n
#   predictions are saved in the predictions folder as predicted.csv and pv_predicted_{Date}.csv
#   @param df:pd.DataFrame - Dataframe to normalize
#   @param model:keras.models - model to be used for the the prediction
#   @param look_back:int - timesteps to look back
#   @param target:string - feature to predict
#   @return df:pd.DataFrame - processed dataframe
def predict_pv_power(df:pd.DataFrame, model, look_back:int=24, target:str="PV power")->None:
  df_pv = df.drop(columns=["Wind speed", "Wind power", "Time"])
  df_future = pd.read_csv(f"{weather_dir}future.csv")
  df_future = keep_next_24_hours_data(df_future, seconds)
  
  pv_future = df_future.drop(columns=["Wind speed", "Wind power"])
  pv_future_norm = add_day_sin_cos_and_normalize(pv_future)
  
  try: 
    for i in range(0, len(pv_future_norm)):
      pv_future_norm[target][i] = predict_next_hour(df_pv, model, look_back, target)
      df_pv = df_pv.append(pv_future_norm.iloc[i])
  except:
    log("PV power prediction process failed")
    
  df_predicted = reverse_normalize(pv_future_norm, target)
  pv_future[target] = df_predicted[target]
  pv_future = pv_future.drop(columns=["Seconds","Day sin","Day cos"])
  
  try:
    pv_future.to_csv(f"{prediction_dir}predicted.csv", index=False)
    log("Predicted data saved to ./data/predictions/predicted.csv")
    pv_future.to_csv(f"{prediction_dir}pv_predicted_{today[:-6]}.csv", index=False)
  except:
    log("Failed to save predicted data")

## Load model and Data and start the PV power prediction
# @return None
def forecast_PV_power()->None:
    log(f"ROOT DIR = {ROOT_DIR}")

    try:
        df_predict = pd.read_csv(processed_data_dir + "make_predictions.csv")
        log("norm loaded")
    except:
        log(f"Unable to load {processed_data_dir}/make_predictions.csv")
        sys.exit(1)

    try:
        pv_model = load_model_from_json("pv_model")
        log(f"pv_model loaded")
    except:
        log("unable to load pv_model")
        sys.exit(1)

    predict_pv_power(df_predict, pv_model, look_back=24, target="PV power")

if __name__ == '__main__':
    forecast_PV_power()