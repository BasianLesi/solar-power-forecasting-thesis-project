import datetime
from itertools import count
import sys
import os
import time
from typing import Union
from config import *
from visualization import *
import pandas as pd
import numpy as np
import math
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from dotenv import find_dotenv, load_dotenv
from sklearn.metrics import mean_squared_error as mse

from keras.models import model_from_json
from keras.models import Sequential, save_model, model_from_json
from keras.layers import *
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras_tuner.tuners import BayesianOptimization
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae 
from sklearn.metrics import r2_score
from keras.utils.vis_utils import plot_model
from tensorflow.keras.optimizers import Adam

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Dataset:
    def __init__(self, name:str, look_back:int=24, target:str="PV power"):
        self.name = name
        self.df = pd.read_csv(processed_data_dir + name + ".csv")
        self.look_back = look_back
        self.target = target
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.split_data()
    
    def split_data(self, train_size:float=0.8, val_size:float=0.1):
        X, y = df_to_input_X_y(self.df,self.look_back, self.target)
        data_size = X.shape[0]
        train_size = math.floor(data_size*0.8)
        val_size = math.floor(data_size*0.1)
        self.X_train, self.y_train = X[:train_size], y[:train_size]
        self.X_val, self.y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
        self.X_test, self.y_test = X[train_size+val_size:], y[train_size+val_size:]
                       

class Model:
    def __init__(self, name:str, dataset:Dataset, num_of_epochs:int=40, num_of_units:int=64):
        self.name = name
        self.model_dir = model_dir + name
        self.dataset = dataset
        self.df = dataset.df
        self.look_back = dataset.look_back
        self.num_of_epochs = num_of_epochs
        self.num_of_units = num_of_units
        self.target = dataset.target
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.trained_model = None
        self.model_history = None

    def train(self)->None:
        model = Sequential()
        model.add(LSTM(self.num_of_units,
                       input_shape=(self.look_back, len(self.df.columns)), 
                       return_sequences=True,  
                       recurrent_dropout=0.1))
        model.add(LSTM(self.num_of_units, 
                       input_shape=(self.look_back, len(self.df.columns)), 
                       return_sequences=False,  
                       recurrent_dropout=0.1))
        model.add(Dropout(0.1))
        model.add(Dense(8, 'relu'))
        model.add(Dense(1, 'relu'))
        model.summary()
        
        create_directory_if_missing(model_dir)
        
        checkpoint = ModelCheckpoint(model_dir, save_best_only=True)
        model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
        
        tensorboard_dir = f"{log_dir}{self.target}_epochs={str(self.num_of_epochs)}_lookback={str(self.look_back)}_{self.timestamp}"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir, histogram_freq=1)
        
        if TUNING:
            self.model_history = bayesian_optimization(model, self.dataset.X_train, self.dataset.y_train, self.dataset.X_val, self.dataset.y_val, self.num_of_epochs)
        else:
            self.model_history = model.fit(self.dataset.X_train, 
                                           self.dataset.y_train, 
                                           validation_data=(self.dataset.X_val, self.dataset.y_val), 
                                           epochs=self.num_of_epochs, 
                                           batch_size=64, 
                                           callbacks=[checkpoint, tensorboard_callback])
        
        self.trained_model = model
        
        if SAVE_FIGURES: plot_model(model, to_file=f'{model_dir}/model_architecture/{self.name}.png', show_shapes=True, show_layer_names=True)
        
        self.benchmark_model_on_test_data()
        if VISUAL: plot_model_history(self)
        self.save_model(model)
        return None
    
    def save_model(self, model)->None:
        # serialize model to JSON
        model_json = model.to_json()
        with open(model_dir + self.name + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(model_dir + self.name + ".h5")
        log("Saved model to disk")   
    
    def benchmark_model_on_test_data(self)->None:
        self.num_of_epochs, self.look_back
        pred:str = "Predicted_"+self.target
        actual:str = "Actual_"+self.target
        predictions = self.trained_model.predict(self.dataset.X_test).flatten()
        df_pred = pd.DataFrame(data={pred:predictions, actual:self.dataset.y_test})
        df_pred = reverse_normalize(df_pred, self.target)
        if VISUAL: plot_predictions(df_pred, self)
        
        mse_score = mse(df_pred.iloc[:, 0], df_pred.iloc[:, 1])
        rmse_score = math.sqrt(mse_score)
        r2score = r2_score(df_pred.iloc[:, 0], df_pred.iloc[:, 1])
        mae_score = mae(df_pred.iloc[:, 0], df_pred.iloc[:, 1])
          
        try:
            metrics = pd.read_csv(f"{metrics_dir}metrics_{self.target}.csv")
            metrics = metrics.append({'mse':mse_score, 'rmse':rmse_score, "r2_score":r2score, "mae":mae_score, "epochs":self.num_of_epochs, "units":self.num_of_units}, ignore_index=True)
            metrics.to_csv(f"{metrics_dir}metrics_{self.target}.csv", index=False)
        except:
            log("No metrics file not found, creating new one")
            metrics = pd.DataFrame(np.array([[mse_score, rmse_score, r2score, mae_score, self.num_of_epochs, self.num_of_units]]), 
                                    columns=["mse", "rmse", "r2_score", "mae", "epochs", "units"])  
            metrics.to_csv(f"{metrics_dir}metrics_{self.target}.csv", index=False)
            

    
        log("mse = " + str(mse_score))
        log("rmse = " + str(rmse_score))
        log("r2_scorem = " + str(r2score))
        log("mae = " + str(mae_score))
        
        
      

def create_directory_if_missing(path:str)->None:
  exists = os.path.exists(path)
  if not exists:
    try:
      os.makedirs(path)
      log("directory created successfully: " + path)
    except:
      log("Unable to create directory: " + path)
  else:
    log("directory already exists")
        
def df_to_input_X_y(df:pd.DataFrame, look_back:int=24, target:str = "PV power")->Union[np.array, np.array]:
    y_index = df.columns.get_loc(target)
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np)-look_back):
        row = [r for r in df_as_np[i:i+look_back]]
        X.append(row)
        label = df_as_np[i+look_back][y_index]
        y.append(label)
    return np.array(X), np.array(y)     

def normalize_column(df:pd.DataFrame, col:int = 1, a:int=0, b:int=1)->pd.DataFrame:
    col_name = df.columns[col]
    df[col_name] = (df[col_name] - df[col_name].min())/(df[col_name].max() - df[col_name].min())
    df[col_name] = (b-a)*df[col_name]+a 
    return df 

def reverse_normalize_column(df:pd.DataFrame, target:str, a:int=0, b:int=1)->pd.DataFrame:
    a = pd.read_csv(processed_data_dir + "merged.csv")  # read the dataframe
    x_min = a[target].min() # get the min value
    x_max = a[target].max() # get the max value
    df[target] = df[target]*(x_max-x_min) + x_min # reverse normalize
    return df
  
def reverse_normalize(df:pd.DataFrame, target:str, a:int=0, b:int=1)->pd.DataFrame:
    a = pd.read_csv(processed_data_dir + "merged.csv")  # read the dataframe
    x_min = a[target].min() # get the min value
    x_max = a[target].max() # get the max value
    for i in df.columns:
      df[i] = df[i]*(x_max-x_min) + x_min # reverse normalize
    return df
  
def bayesian_optimization(X_train, y_train, num_of_epochs)->Sequential:
  bayesian_opt_tuner = BayesianOptimization(
                  build_model,
                  objective='mse',
                  max_trials=200,
                  executions_per_trial=1,
                  directory=os.path.normpath('C:/keras_tuning'),
                  project_name='kerastuner_bayesian_poc',
                  overwrite=True)

  bayesian_opt_tuner.search(X_train, y_train,epochs=num_of_epochs, validation_split=0.2,verbose=1)

  bayes_opt_model_best_model = bayesian_opt_tuner.get_best_models(num_models=20)
  count = 0
  
  for model in bayes_opt_model_best_model:
      save_model(model, f"bayesian_optimization_{str(count)}")
      count += 1
  return bayes_opt_model_best_model[0]

def build_model(hp)->Sequential:
    look_back = 24
    model = Sequential()
    model.add(LSTM(units=hp.Int('units1',min_value=4,max_value=128,step=4), 
                   return_sequences=True, 
                   activation='relu',  
                   recurrent_dropout=hp.Choice('recurrent_dropout1', values=[0, 0.1, 0.2, 0.3, 0.4]), 
                   input_shape=(look_back, 5)))
    
    model.add(LSTM(units=hp.Int('units2',min_value=4,max_value=128,step=4),
                   return_sequences=False,
                   activation='relu', 
                   recurrent_dropout=hp.Choice('recurrent_dropout2', values=[0, 0.1, 0.2, 0.3, 0.4]),
                   input_shape=(look_back, 5)))
    
    model.add(Dropout(hp.Choice('dropout', values=[0, 0.1, 0.2, 0.3, 0.4])))
    model.add(Dense(8, 'relu'))
    model.add(Dense(1, "relu"))
    model.compile(loss='mse', metrics=['mse'], optimizer=Adam(
                    hp.Choice('learning_rate', 
                    values=[1e-2, 1e-3, 1e-4])))
   
    return model


    
if __name__ == '__main__':
    dataset1 = Dataset(name = "pv_norm", look_back = 24, target = "PV power")
    dataset2 = Dataset(name = "wp_norm", look_back = 24, target = "Wind power")
    
    model1 = Model("pv_model", dataset1, 40, 64)
    model2 = Model("wp_model", dataset2, 40, 64)
    
    model1.train()
    model2.train()


































