import requests, os
from myutils import parseFlooding
import pandas as pd


def getFlooding():
    """
    Fetches current flooding data.

    Data
    ---------
    `timestamp`: Time of last observation
    `sensor-id`: ID of the sensor
    `sensor-name`: Name of the sensor
    `latitude` & `longitude`: Location of the sensor
    `water-level`: Water level measured in m
    `status`: Status of the sensor (0,1,2 are flooding and 3,4 are maintenance)
    """

    # Fetch flooding data
    endpoint = "https://app.pub.gov.sg/waterlevel/pages/GetWLInfo.aspx"
    params = {"type": "WL"}
    response = requests.get(endpoint, params=params)
    data = parseFlooding(response.content.decode("utf-8"))
    # Append sensor maxima
    sensors = pd.read_csv(os.path.join(__file__, "../floodmax/sensors.csv"))[
        ["sensor-id", "max-level"]
    ]
    data = data.merge(sensors, on="sensor-id")
    data[r"% full"] = round((data["water-level"] / data["max-level"]) * 100, 2)
    # Reorder columns
    cols = data.columns.tolist()
    cols = cols[:-3] + cols[-2:] + [cols[-3]]
    data = data[cols]

    # TODO: Add logging
    return data
