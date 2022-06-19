'''
train_model module - model training and tuning.  
   
    1. Train PV and Wind power forecasting models
    2. Options for bayesian optimizatino tuning
    3. Benchmarking of models
    4. Save models training logs and display using tensorboard
    5. Save models for future use
'''
##
# @package train_model - Module for training and tuning of models
# 
#

import os
import math
import datetime
import numpy as np
import pandas as pd
from config import *
from typing import Union
from visualization import *

from keras.layers import *
from sklearn.metrics import r2_score
from keras.losses import MeanSquaredError
from keras.utils.vis_utils import plot_model
from tensorflow.keras.optimizers import Adam
from keras.metrics import RootMeanSquaredError
from keras.models import Sequential, save_model
from keras_tuner.tuners import BayesianOptimization
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae 
from keras.callbacks import ModelCheckpoint, TensorBoard

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable tensorflow warnings


class Dataset: 
    ## Object initialization
    #   
    #   @param self:Dataset - object instance
    #   @param name:string - name of dataset
    #   @param look_back:int - look back period for model training input shape
    #   @param target:string - target column name for model training   
    def __init__(self, name:str, look_back:int=24, target:str="PV power"):
        self.name:str = name
        self.df:pd.Dataframe = pd.read_csv(processed_data_dir + name + ".csv")
        self.look_back:int = look_back
        self.target:str = target
        self.X_train:np.array = None
        self.y_train:np.array = None
        self.X_val:np.array = None
        self.y_val:np.array = None
        self.X_test:np.array = None
        self.y_test:np.array = None
        self.split_data()
    
    ## split data into train, validation and test sets
    #
    #   @param self:Dataset - object instance
    #   @param t_size:float - train data size (default 0.8)
    #   @param v_size:float - validation data size (default 0.2)
    #   @return None 
    def split_data(self, t_size:float=0.8, v_size:float=0.1)->None:
        X, y = self.df_to_input_X_y()
        data_size = X.shape[0]
        train_size = math.floor(data_size*t_size)
        val_size = math.floor(data_size*v_size)
        self.X_train, self.y_train = X[:train_size], y[:train_size]
        self.X_val, self.y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
        self.X_test, self.y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    ##  Generate input and target numpy arrays from dataframe
    #   @param self:Dataset - object instance
    #   @return np.array, np.array - input and target numpy arrays        
    def df_to_input_X_y(self)->Union[np.array, np.array]:
        y_index = self.df.columns.get_loc(self.target)
        df_as_np = self.df.to_numpy()
        X = []
        y = []
        for i in range(len(df_as_np)-self.look_back):
            row = [r for r in df_as_np[i:i+self.look_back]]
            X.append(row)
            label = df_as_np[i+self.look_back][y_index]
            y.append(label)
        return np.array(X), np.array(y)                

class Model:
    ## Object initialization
    #   
    #   @param self:Model - object instance
    #   @param name:string - model name
    #   @param dataset:Dataset - dataset object instance containing training data
    #   @param num_epochs:int - number of epochs for model training (default 40)
    #   @param num_of_units:int - number of units in hidden layers (default 64)
    def __init__(self, name:str, dataset:Dataset, num_of_epochs:int=40, num_of_units:int=64):
        self.name = name
        self.model_dir = model_dir + name
        self.dataset = dataset
        self.df = dataset.df
        self.look_back = dataset.look_back
        self.num_of_epochs = num_of_epochs
        self.num_of_units = num_of_units
        self.target = dataset.target
        self.timestamp = int(datetime.today().timestamp())
        self.trained_model = None
        self.model_history = None

    ## Model training
    #   
    #   @param self:Model - object instance
    #
    #   training model using Keras Sequential model \n 
    #   two LSTM layers are used for model training \n
    #   one dropout Layer and two dense layers with relu activation \n
    #   trained model is saved along with benchmark metrics and tensorboard logs
    #   @return None
    def train(self)->None:
        log("Training model: " + self.name)
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
        self.save_model()
        return None
    
    ## Save trained model in h5 format
    #   @param self:Model - object instance
    #   @return None
    def save_model(self)->None:
        # serialize model to JSON
        model_json = self.trained_model.to_json()
        with open(model_dir + self.name + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.trained_model.save_weights(model_dir + self.name + ".h5")
        log("Saved model to disk")   
    
    ##  Trained model benchmark on test data
    #   @param self:Model - object instance
    #   results and figures generated are saved in reports directory
    #   @return None
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
            log(f"{metrics_dir}metrics{self.target}.csv file not found, creating new one")
            metrics = pd.DataFrame(np.array([[mse_score, rmse_score, r2score, mae_score, self.num_of_epochs, self.num_of_units]]), 
                                    columns=["mse", "rmse", "r2_score", "mae", "epochs", "units"])  
            metrics.to_csv(f"{metrics_dir}metrics_{self.target}.csv", index=False)
            

    
        log("mse = " + str(mse_score))
        log("rmse = " + str(rmse_score))
        log("r2_scorem = " + str(r2score))
        log("mae = " + str(mae_score))
        
        
      
##  Helper function - Creates directory if it does not exist
#   @param path:string - path to directory
#   @return:None
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
    
##  Normalize target value - used by benchmark for the predicted and actual values
#   @param df:DataFrame - dataframe with target value
#   @param target:string - target column name
#   @return DataFrame - normalized dataframe
def reverse_normalize(df:pd.DataFrame, target:str)->pd.DataFrame:
    a = pd.read_csv(processed_data_dir + "merged.csv")  # read the dataframe
    x_min = a[target].min() # get the min value
    x_max = a[target].max() # get the max value
    for i in df.columns:
      df[i] = df[i]*(x_max-x_min) + x_min # reverse normalize
    return df

##  Bayesian optimization - used to find the best parameters for the model (20 best models are saved)
#   @param X_train:numpy.ndarray - training data
#   @param y_train:numpy.ndarray - training target
#   @param num_of_epochs:int - number of epochs
#   @return Sequential - trained model 
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

##  Bayesian optimization helper function hypermodel(hp) constructor
#   @param hp:Hypermodel - Hypermodel object
#   @return Sequential - trained model 
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