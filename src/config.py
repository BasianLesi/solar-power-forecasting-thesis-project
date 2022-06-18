from pathlib import Path

DEBUG = False # Set True to print debug messages
VISUAL= False # Set True to plot and save figures
             

ROOT_DIR = str(Path(__file__).parent.parent)  #Project root directory

# Useful frequently used paths
raw_data_dir = ROOT_DIR+"/data/raw/"    
processed_data_dir = ROOT_DIR+"/data/processed/" 
weather_dir = ROOT_DIR+"/data/weather/"
model_dir = ROOT_DIR+"/models/"
figures_dir = ROOT_DIR+"/reports/figures/"
metrics_dir = ROOT_DIR+"/reports/metrics/"

def log(s):
    if DEBUG:
        print(s)
