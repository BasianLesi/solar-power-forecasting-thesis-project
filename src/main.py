##
#    @mainpage Implementation of Deep Learning Algorithm for Solar Forecasting on Cloud/IoT Platform
#
#    @authors                 Basian Lesi

##
# @package main - main pipeline function to run on loop
#

from config import *
from data_fetching import *
from upload_to_cloud import *
from predict_model import *
import time

sec = 60
min = 60
hour = float(sec*min) # data is updated every hour

if __name__ == '__main__':
    starttime = time.time()
    while True:
        log("task scheduled to run every hour")
        make_predictions_data()
        forecast_PV_power()
        upload_to_google_sheets()
        log("sleep for an hour")
        time.sleep(hour- ((time.time() - starttime) % hour))