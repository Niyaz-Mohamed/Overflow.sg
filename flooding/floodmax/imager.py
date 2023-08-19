import os
import pandas as pd
import pyautogui as auto
from time import time

# Position list/Settings
auto.PAUSE = 1
SEARCHBAR = (1340, 85)
DRAGLOCATION = (0, 700)
SCROLLDIST = 1380
IMAGETOP = (650, 300)
# With reference to top right
IMAGEWIDTH = 1500 - IMAGETOP[0]
# With reference to bottom left
IMAGEHEIGHT = 800 - IMAGETOP[1]

# Fetch stations
sensors = pd.read_csv(os.path.join(__file__, "../stations.csv"))
sensorCount = 0
timer = time()
for index, sensor in sensors.iterrows():
    url = (
        "https://app.pub.gov.sg/waterlevel/pages/WaterLevelReport.aspx?stationid="
        + sensor["sensor-id"]
    )

    # Search the url
    auto.PAUSE = 0.6
    auto.moveTo(*SEARCHBAR)
    auto.click()
    auto.press("backspace")
    auto.write(url)
    auto.press("enter")

    # Scroll down slightly
    auto.moveTo(*DRAGLOCATION)
    auto.PAUSE = 1
    auto.scroll(-SCROLLDIST)

    # Take a screenshot
    auto.screenshot(
        f"{sensor['sensor-id']}.png",
        region=(*IMAGETOP, IMAGEWIDTH, IMAGEHEIGHT),
    )

    # Log details
    sensorCount += 1
    timeSpent = round(time() - timer, 2)
    print(
        f"Chart-{index}\nTime spent: {round(time()-timer)}s\nAv. time per chart: {round(timeSpent / sensorCount)}\n"
    )

auto.alert("COMPLETED EXTRACTION")
