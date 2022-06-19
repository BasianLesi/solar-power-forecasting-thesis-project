##
# @package upload_to_cloud - Upload PV power forecasting to google sheets
#

import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pprint import pprint
import pandas as pd
from config import *

##  Uplaod predicted.csv data to the cloud
#   @param hours:int - hours of future predictions to be uploaded
#   @return None
def upload_to_google_sheets(hours:int=24)->None:
    
    if os.path.exists(f"{config_dir}creds.json") == False:
        print("Error in upload_to_cloud.py -- creds.json file does not exist")
        print("Get credencials file from: https://console.developers.google.com")
        exit(1)
    
    scope = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
    ]  
    creds = ServiceAccountCredentials.from_json_keyfile_name(f"{config_dir}creds.json", scope)  
    client = gspread.authorize(creds)
    try:
        sheet = client.open("future").sheet1 
    except:
        log("unable to open google sheet")
        exit(1)

    df = pd.read_csv(prediction_dir + 'predicted.csv')
    for i in range(hours-1, -1, -1):
        insertRow = df.iloc[i].values.flatten().tolist()
        for s in range(1, len(insertRow)):
            insertRow[s] = float(insertRow[s])
        
        sheet.insert_row(insertRow, 2)
    pass        
    sheet.delete_rows(26, 26+hours)
    log("updated google sheet successfully")

if __name__ == '__main__':
    upload_to_google_sheets()