import os
import pandas as pd
import pyautogui as auto

# Position list/Settings
auto.PAUSE = 0.5
SEARCHBAR = (1340, 85)
DRAGLOCATION = (0, 700)
SCROLLDIST = 1380
IMAGETOP = (650, 310)
# With reference to top right
IMAGEWIDTH = 1490 - IMAGETOP[0]
# With reference to bottom left
IMAGEHEIGHT = 690 - IMAGETOP[1]

# Fetch stations
sensors = pd.read_csv(os.path.join(__file__, "../stations.csv"))
for index, sensor in sensors.iterrows():
    url = (
        "https://app.pub.gov.sg/waterlevel/pages/WaterLevelReport.aspx?stationid="
        + sensor["sensor-id"]
    )

    # Search the url
    auto.moveTo(*SEARCHBAR)
    auto.click()
    auto.press("backspace")
    auto.write(url)
    auto.press("enter")

    # Scroll down slightly
    auto.moveTo(*DRAGLOCATION)
    auto.scroll(-SCROLLDIST)

    # Take a screenshot
    auto.screenshot(
        f"{sensor['sensor-id']}.png",
        region=(*IMAGETOP, IMAGEWIDTH, IMAGEHEIGHT),
    )
