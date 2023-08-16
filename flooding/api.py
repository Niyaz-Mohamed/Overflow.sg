import requests
from myutils import parseFlooding


def getFlooding():
    """
    Fetches current flooding data.

    Data
    ---------
    `timestamp`: Time of last observation
    `sensor-id`: ID of the sensor
    `sensor-name`: Name of the sensor
    `latitude` & `longitude`: Location of the sensor
    `water-level`: Water level measured in mm
    `status`: Status of the sensor (0,1,2 are flooding and 3,4 are maintenance)
    """
    endpoint = "https://app.pub.gov.sg/waterlevel/pages/GetWLInfo.aspx"
    params = {"type": "WL"}
    response = requests.get(endpoint, params=params)
    data = parseFlooding(response.content.decode("utf-8"))
    return data.sort_values(by=["timestamp", "sensor-id"])
