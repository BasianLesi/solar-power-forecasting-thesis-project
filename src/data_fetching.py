'''
Data fetching module:  

    1. Fetch Weather Data
        i.   past - used as past datapoints in the forecasting model
        ii.  present - data at present time
        iii. future - weather forecasting for the next 48 hours
        
    2. Preprocessing:
        i.   Merge past, present and future data into one dataframe
        ii.  Extract features and normalize
        iii. Save data for predictions
'''

##
# @package data_fetching - Get weather forecasting data required as input to the model for the forecasting process.
# 
#
import pandas as pd
import numpy as np
from datetime import date,datetime
import requests
import json
from config import *

##  Update global time and time dependent variables
#   
#   @return None
def update_global_variables()->None: 
    global today, seconds, tomorrow,yesterday, today_url, yesterday_url, two_day_forecast_url
    today = date.today()
    seconds = int(datetime.today().timestamp())
    tomorrow = seconds + day;
    yesterday = seconds - day;
    today = datetime.fromtimestamp(seconds).strftime("%d-%m-%Y %H:%M")

    yesterday_url = f"https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={lon}&dt={yesterday}&appid={api_key}&units=metric"
    today_url     = f"https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={lon}&dt={seconds}&appid={api_key}&units=metric"
    two_day_forecast_url = f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude=alerts&appid={api_key}&units=metric"
    
##  API call to OpenWeather.com 
#   @param url:string - the url for the api request 
#   @return pd.DataFrame - processed Dataframe with api response features of interest
def weather_api_call(url:str)->pd.DataFrame:    
    response = requests.get(url)    # Make a call to the api
    forecast = json.loads(response.text)    # Convert the response to json
    time = []   # Create a list to store the time
    temperature = []    # Create a list to store the temperature
    uvi = []    # Create a list to store uv index
    wind = []   # Create a list to store wind
    power = []  # Create a list to store power

    for i in range(0, len(forecast["hourly"])):   # Loop through the json response
        ts = forecast["hourly"][i]["dt"]    # Get the time
        #date_time = datetime.utcfromtimestamp(ts).strftime('%d-%m-%Y %H:%M')    # Convert the time to a string
        date_time = datetime.fromtimestamp(ts).strftime("%d-%m-%Y %H:%M")
        time.append(date_time)  # Append the time to the list
        temperature.append(forecast["hourly"][i]["temp"])   # Append the temperature to the list
        uvi.append(forecast["hourly"][i]["uvi"]*100)    # Append the uvi to the list
        wind.append(forecast["hourly"][i]["wind_speed"])    # Append the wind to the list
        power.append(0)   # Append the power to the list

    df = pd.DataFrame(data={"Time":time, "Temperature":temperature, "PV power":power, "Solar radiation":uvi, "Wind power":power, "Wind speed":wind})
    return df

# Returns a url string for the specified seconds
def generate_url(_seconds:int)->str:
    return f"https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={lon}&dt={_seconds}&appid={api_key}&units=metric"


##  Get past weather data and save to past.csv
#   @return None
def get_past_weather()->None:
    days = []   # List of days
    seconds = int(datetime.today().timestamp()) # Current seconds
    for i in range(1,6):    # Get the past 5 days
        days.append(seconds-i*day)  # Add the seconds to the list
    df_list = []    # List of dataframes
    for sec in days:    # For each day
        df_list.append(weather_api_call(generate_url(sec)))     # Get the dataframe for that day
    df_concat = pd.concat(df_list)  # Concatenate the dataframes
    df_concat = df_concat.sort_values(by='Time')    # Sort the dataframe by time
    df_concat.drop_duplicates(subset = ['Time'], keep = 'first', inplace = True) # Remove duplicates
    
    try:
        df = pd.read_csv(f"{weather_dir}past.csv")  # Read the past dataframe
    except:
        log("{weather_dir}past.csv not found, creating new one")
        df = pd.DataFrame(columns=["Time", "PV power", "Solar radiation", "Wind power", "Wind speed"])  
        df.to_csv(f"{weather_dir}past.csv", index=False)
        df = pd.read_csv(f"{weather_dir}past.csv")  # Read the past dataframe
    
    df_concat = pd.concat([df, df_concat])  # Concatenate the dataframes
    df_concat.index = pd.to_datetime(df_concat['Time'], format='%d-%m-%Y %H:%M')    # Set the index to time
    df_concat['Seconds'] = df_concat.index.map(pd.Timestamp.timestamp)  # Set the seconds column
    df_concat.drop_duplicates(subset = ['Seconds'], keep = 'first', inplace = True) # Remove duplicates
    df_concat = df_concat.sort_values(by='Seconds') # Sort by seconds
    df = df_concat.drop('Seconds', axis=1)  # Drop the seconds column
    df.drop_duplicates(subset = ['Time'], keep = 'first', inplace = True) # Remove duplicates
    df.to_csv(f"{weather_dir}past.csv", index=False)    # Save the dataframe

##  Get future weather forecasting data and save to future.csv
#   @return None
def get_future_weather()->None:
    day = 48 #hours
    df = weather_api_call(two_day_forecast_url)
          
    try:
        df1 = pd.read_csv(f"{weather_dir}future.csv")  # Read the past dataframe
    except:
        log("{weather_dir}future.csv not found, creating new one")
        df1 = pd.DataFrame(columns=["Time", "PV power", "Solar radiation", "Wind power", "Wind speed"])  
        df1.to_csv(f"{weather_dir}past.csv", index=False)
        
    df_concat = pd.concat([df[:day], df1])
    df_concat.index = pd.to_datetime(df_concat['Time'], format='%d-%m-%Y %H:%M')
    df_concat['Seconds'] = df_concat.index.map(pd.Timestamp.timestamp)
    df_concat.drop_duplicates(subset = ['Seconds'], keep = 'first', inplace = True)
    df_concat = df_concat.sort_values(by='Seconds')
    df = df_concat.drop('Seconds', axis=1)
    df.to_csv(f"{weather_dir}future.csv", index=False)

##  Get present weather data and save to future.csv
#   @return None
def get_present_weather()->None:
    update_global_variables()
    df = weather_api_call(today_url)
    df.to_csv(f"{weather_dir}present.csv", index=False)
    
##  Update past, present and future data
#   @return None
def update_data()->None:
    ##TODO: Fix: if past.csv is not found should call get_past_weather() twice!!
    get_past_weather()   
    get_future_weather()
    get_present_weather()
    get_past_weather()
    log("data updated successfully")

##  Normalize dataframe column on range[a,b]
#   @param df:pd.DataFrame - Dataframe to normalize
#   @param col:int - Index to column
#   @param a:int - Min value (default 0)
#   @param b:int - Max value (default 1)
#   @return pd.DataFrame - Normalized dataframe
def normalize_column(df_forecast:pd.DataFrame, col:int = 1, a:int=0, b:int=1)->pd.DataFrame:
    df = pd.read_csv(processed_data_dir + "merged.csv")
    col_name = df_forecast.columns[col]
    max = df[col_name].max()
    min = df[col_name].min()
    df_forecast[col_name] = (df_forecast[col_name] - min)/(max - min)
    df_forecast[col_name] = (b-a)*df_forecast[col_name]+a 
    return df_forecast 

##  Process and normalize fetched data to be ready for forecasting model input.
#   @return None
def make_predictions_data()->None:
    update_data()
    df_past = pd.read_csv(f"{weather_dir}past.csv")
    df_present = pd.read_csv(f"{weather_dir}present.csv")
    df_future = pd.read_csv(f"{weather_dir}future.csv")
    df_past_present = pd.concat([df_past, df_present])
    x = len(df_past_present)
    df = df_past_present[x-24:]
    df.to_csv(f"{processed_data_dir}preprocessed.csv", index=False)

    # feature extraction and normalization
    df.index = pd.to_datetime(df['Time'], format='%d-%m-%Y %H:%M')
    df['Seconds'] = df.index.map(pd.Timestamp.timestamp)
    day = 60*60*24
    df['Day sin']  = np.sin(df['Seconds'] * (2 * np.pi / day))
    df['Day cos']  = np.cos(df['Seconds'] * (2 * np.pi / day))
    df = df.drop('Seconds', axis=1)
    for i in range (1,len(df.columns)):
        df = normalize_column(df, i)
    df.to_csv(f"{processed_data_dir}make_predictions.csv", index=False)


if __name__ == '__main__':
    update_data()
    make_predictions_data()