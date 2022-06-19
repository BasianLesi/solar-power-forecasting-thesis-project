##
# @package visualization - Visualization functions
# 
#

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from config import *

## Plots model predictions vs actual values
#
#   @param df_pred:pd.Dataframe - dataframe with the actual and predicted values
#   @param model:Model  - Model object
#
#   @note - Generated plots are saved in figures_dir specified in config.py module
#
#   @return None 
def plot_predictions(df_pred:pd.DataFrame, model, start:int=0, end:int=400):
  pred:str = "Predicted_"+model.target
  actual:str = "Actual_"+model.target
  plt.plot(df_pred[pred][start:end], "b", label="Predicted")
  plt.plot(df_pred[actual][start:end], "tab:orange", label="Actual", alpha=0.8)
  plt.legend(loc='upper left')
  mse_score = mse(df_pred[pred], df_pred[actual])
  plt.title("Actual vs Predicted \n mse = " + str(round(mse_score,3)))
  plt.ylabel(f"{model.target} (MW)")
  plt.xlabel("Time (Hours)")
  if SAVE_FIGURES: plt.savefig(f"{figures_dir}_Prediction_vs_Actual_{model.name}_{model.timestamp}.png", format='png', dpi=300)
  plt.show()
  plt.clf() 
  
## Plots model history - Training vs Validation loss and RMSE
#
#   @param df_pred:pd.Dataframe - dataframe with the actual and predicted values
#   @param model:Model  - Model object
#
#   @note - Generated plots are saved in figures_dir specified in config.py module
#
#   @return None 
def plot_model_history(model):
    plt.plot(model.model_history.history['loss'], 'g', label='Training loss')
    plt.plot(model.model_history.history['val_loss'], 'b', label='Validation loss')
    df = pd.DataFrame(data={"Training_loss":model.model_history.history['loss'], "Validation_loss":model.model_history.history['val_loss']})
    plt.title('Training and Validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='upper right')
    if SAVE_FIGURES: plt.savefig(f"{figures_dir + model.name}_loss_{model.timestamp}.png", format='png', dpi=300)
    plt.show()
    plt.clf()
    
    plt.plot(model.model_history.history['root_mean_squared_error'], 'g', label='Training RMSE')
    plt.plot(model.model_history.history['val_root_mean_squared_error'],'b', label='Validation RMSE')
    df = pd.DataFrame(data={"Training_RMSE":model.model_history.history['root_mean_squared_error'], "Validation_RMSE":model.model_history.history['val_root_mean_squared_error']}) 
    plt.title('Training and Validation RMSE')
    plt.ylabel('RMSE')
    plt.xlabel('Epochs')
    plt.legend(loc='upper right')
    if SAVE_FIGURES: plt.savefig(f"{figures_dir + model.name}_RMSE_{model.timestamp}.png", format='png', dpi=300)
    plt.show()
    plt.clf()