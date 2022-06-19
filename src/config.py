'''
Project configuration module:  

    1. Settings
        i.   DEBUG - Set to print debug messages
        ii.  VISUAL - Set True to plot
        iii. TUNING - Set True to for hyperparameter tuning using bayesian optimization 
        iv.  SAVE_FIGURES - Set True to save figures
        
    2. Directories:
        i. raw_data_dir - Directory or raw data
        ii. processed_data_dir - Directory of processed data
        iii. weather_data_dir - Directory of weather data
        iv.  model_dir - Directory of saved models
        v. figures_dir - Directory of saved figures
        vi. metrics - Directory of models metric
        vii. log_dir - Directory of tensorboard logs
 
    3. log(s) - modified print() function
'''

##
# @package config - Project configurations (root project and useful directories), project settings modes (debug, visual, tuning, save_figures)
# 
#
from datetime import date,datetime
from pathlib import Path
import json
import os
import warnings



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress tensorflow warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # disable GPU

#global variables
DEBUG = False # Set True to print debug messages
VISUAL = False # Set True to plot
TUNING = False # Set True to for hyperparameter tuning using bayesian optimization            
SAVE_FIGURES = False # Set True to save figures

if DEBUG: warnings.filterwarnings('ignore') # surress warning

# check if is running on Windows and set project root path accordingly
if os.name == 'nt':
    ROOT_DIR = str(Path(__file__).parent.parent) 
else:
    ROOT_DIR = str(Path(os.path.abspath(os.curdir)))

# useful frequently used directories
raw_data_dir = ROOT_DIR+"/data/raw/" 
processed_data_dir = ROOT_DIR+"/data/processed/" 
weather_dir = ROOT_DIR+"/data/weather/"
prediction_dir = ROOT_DIR+"/data/predictions/"
model_dir = ROOT_DIR+"/models/"
figures_dir = ROOT_DIR+"/reports/figures/"
metrics_dir = ROOT_DIR+"/reports/metrics/"
log_dir = ROOT_DIR+"/models/tensorboard_logs/"
config_dir = ROOT_DIR+"/config/"

# global time variables in seconds to be used for api calls
day = 60*60*24  
today = date.today()  
seconds = int(datetime.today().timestamp()) 
tomorrow = seconds + day;   
yesterday = seconds - day;  
today = datetime.fromtimestamp(seconds).strftime("%d-%m-%Y %H:%M")

# OpenWeather api variables and url requests
api_key = ""
lat = ""
lon = ""
yesterday_url = ""
today_url     = ""
two_day_forecast_url = ""


##  Read OpenWeather_API_settings.json file and set global variables
#   @return None
def load_config_files()->None:
    global api_key, lat, lon, yesterday_url, today_url, two_day_forecast_url
    with open(f'{config_dir}OpenWeather_API_settings.json', 'r') as f:
        config = json.load(f)

    if config["api_key"] == "":
        print("Missing api_key on config/OpenWeatehr_API_settings.json file")
        exit(1)
        
    if config["lat"] == "" or config["lon"] == "":
        print("Missing latitude or longitude on config/OpenWeatehr_API_settings.json file")
        exit(1)
        
    api_key = config["api_key"]
    lat = config["lat"]
    lon = config["lon"]
    yesterday_url = f"https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={lon}&dt={yesterday}&appid={api_key}&units=metric"
    today_url     = f"https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={lon}&dt={seconds}&appid={api_key}&units=metric"
    two_day_forecast_url = f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude=alerts&appid={api_key}&units=metric"


## Modified python print function to check for DEBUG flag before printing to console
#   @param s:string - String to be printed out to console
#   @return None
def log(s)->None:
    if DEBUG:
        print(s)


load_config_files()