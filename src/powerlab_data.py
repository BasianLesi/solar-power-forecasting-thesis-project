#this implementation currently works on unix based systems and requires selenium firefox webdriver

import time
import pandas as pd
from lxml import html
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.firefox.options import Options

# We don't want to open the webpage in a real browser, but in a headless browser.
def fetch_actual_pv_data(options, driver):
# driver.get("http://quotes.toscrape.com/js/")
  driver.get("https://bornholm.powerlab.dk/")
  page_source = driver.page_source
  tree = html.fromstring(page_source)
  id = tree.xpath('//div[@id="sub_solar_cells"]')

  value = id[0].xpath('//div[@class="value"]/text()')
  power = float(value[2][:-2])
  
  time = int(datetime.today().timestamp())

  metrics = pd.read_csv("actual_pv_power.csv")
  metrics = metrics.append({"Time":time, "PV power":power}, ignore_index=True)
  metrics.to_csv("actual_pv_power.csv", index=False)


if __name__ == '__main__':
    starttime = time.time()
    options = Options()
    options.add_argument('--headless')
    driver = webdriver.Firefox(options=options)

    while True:
        #fetch solar power data form https://bornholm.powerlab.dk every minute
        fetch_actual_pv_data(options, driver)
        time.sleep(60.0 - ((time.time() - starttime) % 60.0))