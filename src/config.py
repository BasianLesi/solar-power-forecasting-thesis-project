from pathlib import Path

DEBUG = True # Set True to print debug messages
VISUAL = True # Set True to plot
TUNING = False # Set True to for hyperparameter tuning using bayesian optimization            
SAVE_FIGURES = False # Set True to save figures

ROOT_DIR = str(Path(__file__).parent.parent)  #Project root directory

# Useful frequently used paths
raw_data_dir = ROOT_DIR+"/data/raw/"    
processed_data_dir = ROOT_DIR+"/data/processed/" 
weather_dir = ROOT_DIR+"/data/weather/"
model_dir = ROOT_DIR+"/models/"
figures_dir = ROOT_DIR+"/reports/figures/"
metrics_dir = ROOT_DIR+"/reports/metrics/"
log_dir = ROOT_DIR+"/models/tensorboard_logs/"

def log(s):
    if DEBUG:
        print(s)
