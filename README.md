# Thesis Project
> Implementation of Deep Learning Algorithm for Solar Forecasting on Cloud/IoT Platform.
<!-- > Live demo [_here_](https://www.example.com). If you have the project hosted somewhere, include the link here. -->

## Table of Contents
- [](#)
  - [Table of Contents](#table-of-contents)
  - [General Information](#general-information)
  - [Features](#features)
  - [Setup Raspberry Pi Environment](#setup-raspberry-pi-environment)
  - [Setup Windows or Linux Environment](#setup-windows-or-linux-environment)
  - [Obtain API keys](#obtain-api-keys)
    - [OpenWeather API key](#1-openweather-api-key)
    - [Google Sheets API key](#2-google-sheets-api-key)
  - [Usage](#usage)
  - [Documentation](#documentation)
  - [Project Status](#project-status)
  - [Room for Improvement](#room-for-improvement)
  - [Contact](#contact)
<!-- * [License](#license) -->


## General Information
- Solar power forecasting pipeline (Data acquisition, processing, forecasting and publishing). 
- Training an LSTM model for solar power forecasting, (training is not performed on Raspberry Pi).
- LSTM model is used to predict the solar power output for the next 24 hours.
- Predictions are uploaded to cloud (Google sheets).
- The project is hosted on a Raspberry Pi model 3b

## Setup Windows or Linux Environment
To set up on Windows or Linux machine, in case of additionally training the model. 
- Create a python virtual environment.
- install requirements `pip install -r requirements.txt`. 

## Setup Raspberry Pi Environment
It is recommended to install the specified versions, for compatibility reasons with RPi's cpu ARM architecture.



### 1. Install recommended OS, python and Tensorflow versions
- Raspberry pi OS. 
  - **Debian version: 10 (buster)** : https://downloads.raspberrypi.org/raspios_lite_armhf/images/raspios_lite_armhf-2021-01-12/.
- python 3.7.3 (By default included in Debian 10 (buster)).
- Tensorflow - 2.4.0: https://github.com/lhelontra/tensorflow-on-arm/releases.

### 2. Install required libraries specified in requirements.txt file:
```console
pip install -r requirements.txt
```

### 2.1 - If numpy installation error occurs follow this steps:
```console
sudo apt-get update\
pip install cython
sudo apt-get install gcc python3.7-dev
pip install pycocotools==2.0.0
pip install --upgrade numpy==1.20.1
```

### 2.2 - Same if h5py installation error occurs:
```console
pip uninstall h5py\
sudo apt-get install libhdf5-dev\
pip install h5py
pip install --upgrade h5py==2.10.0
```

### 3. Modify Tensorflow function
To overcome this issue: https://github.com/tensorflow/models/issues/9706.

by following glemarivero's suggestion:\
modify **_constant_if_small(value, shape, dtype, name)** function: 

```console
sudo vim /home/pi/tensorflow/lib/python3.7/site-packages/tensorflow/python/ops/array_ops.py
```
replace  `np.prod(shape)` with `reduce_prod(shape)`, and \
import `from tensorflow.math import reduce_prod` : the modified code should look like this:

```python
from tensorflow.math import reduce_prod # we import this on top of the file
def _constant_if_small(value, shape, dtype, name):\
  try:\
    if np.prod(shape) < 1000: ## --> it to if reduce_prod(shape) < 1000:\
      return constant(value, shape=shape, dtype=dtype, name=name)\
  except TypeError:\
    # Happens when shape is a Tensor, list with Tensor elements, etc.\
    pass\
  return None
  ```
## Obtain API keys
This project uses the below API's:
  - https://openweathermap.org/ - to obtain weather data.
  - https://console.cloud.google.com/ - Google Sheets where the forecasted data will be uploaded and displayed.

### 1. OpenWeather API key
1. Go to https://home.openweathermap.org/api_keys.
1. `Create key` -> insert the key name and press `Generate`.
2. Copy the generated key and add it on the existing [OpenWeather_API_settings.json](https://github.com/BasianLesi/solar-power-forecasting-thesis-project/blob/master/config/OpenWeather_API_settings.json#:~:text=%7B-,%22api_key%22%3A%20%22%22,-%2C%20%22lat%22) file.  
3. The final file file should looks like below:

```json
{"api_key": "xxxxxxxxxxxxxxx", "lat": "55.1449", "lon": "14.9170"}
```

### 2. Google Sheets API key
The aim is to create API keys for Google Sheets access and obtain credentials creds.json file similar to below:
```json
{
    "type": "service_account",
    "project_id": "testsheets-xxxxxx",
    "private_key_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "private_key": "-----BEGIN PRIVATE KEY-----\nxxxxxxxxxxxxxxxxxxxxxxx\n-----END PRIVATE KEY-----\n",
    "client_email": "example@testsheets-xxxxx.iam.gserviceaccount.com",
    "client_id": "xxxxxxxxxxxx",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/example%40testsheets-xxxxxx.iam.gserviceaccount.com"
  }
```
1. Go to https://console.cloud.google.com/ Dashboard and CREATE PROJECT (follow the set up steps).
1. Navigation `button` -> `APIs & Services` -> `ENABLE APIS AND SERVICES`.
1. Select `Google Sheets API` -> `ENABLE`.
1. Go to Credentials tab -> `CREATE CREDENTIALS` -> `Service account` (follow the set up steps).
1. On Credentials tab -> click the email on Service Accounts section `example@testsheets-xxxxx.iam.gserviceaccount.com`.
1. On Keys section click `ADD KEY` -> `Create new key` -> `JSON` -> `creds.json` file will be downloaded (rename to creds.json if necessary).
1. Create a new Google spreadsheet https://docs.google.com/spreadsheets/.
1. On the created spreadsheet page click `Share` -> add `example@testsheets-xxxxx.iam.gserviceaccount.com` (with editor option) -> and press `Share`. 
1. Add creds.json file to the project [./config/](https://github.com/BasianLesi/solar-power-forecasting-thesis-project/tree/master/config) directory.


## Usage
After successfully completing the above steps, to start the pipeline run:

`python ./src/main.py`

Now the pipeline should be up and running. It should now successfully make solar power generation forecasting for the next 24 hours and upload to cloud. The forecasting is updated hourly.

#### Additional settings
In [src/config.py](https://github.com/BasianLesi/solar-power-forecasting-thesis-project/blob/master/src/config.py#:~:text=DEBUG%20%3D%20True,to%20save%20figures) file we can modify the settings below:
```python
DEBUG = False # Set True to print debug messages
VISUAL = False # Set True to plot
TUNING = False # Set True to for hyperparameter tuning using bayesian optimization            
SAVE_FIGURES = False # Set True to save figures
```

## Documentation
Analytical source code documentation can be viewed [in this pdf file](https://github.com/BasianLesi/solar-power-forecasting-thesis-project/blob/master/doxygen_docs/documentation.pdf) or for an [this html version](https://github.com/BasianLesi/solar-power-forecasting-thesis-project/blob/master/doxygen_docs/html/) launch [this html version](https://github.com/BasianLesi/solar-power-forecasting-thesis-project/blob/master/doxygen_docs/html/index.html) .

## Project Status
_in progress_ 


## Room for Improvement
- Develop a robust reliable source for fetching Solar and Wind power generated data for bornholm area past input in the model:
- Improve Wind power model by adding additional features weather features (wind direction)

## Contact
Created by [@BasianLesi](basian.lesi@gmai.com) - feel free to contact me!