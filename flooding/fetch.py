from datetime import datetime
import requests
import pandas as pd


def getFlooding(dt: datetime):
    endpoint = "https://app.pub.gov.sg/waterlevel/pages/GetWLInfo.aspx"
    params = {"type": "WL", "d": dt}
    response = requests.get(endpoint, params=params)
    return response.content.decode("utf-8")


def parseTimestamp(timestamp: str):
    """
    Parses timestamps, accounting for possible errors in system
    """
    # Standardise timestamp, zero pad all
    timestamp = timestamp.split(" ")
    while "" in timestamp:
        timestamp.remove("")
    # Pad day
    if len(timestamp[1]) < 2:
        timestamp[1] = "0" + timestamp[1]
    # Pad time
    if len(timestamp[-1].split(":")[0]) < 2:
        timestamp[-1] = "0" + timestamp[-1]
    timestamp = " ".join(timestamp)
    timestamp = datetime.strptime(timestamp, "%b %d %Y  %I:%M%p")
    return timestamp


def parseFlooding(raw: str):
    """
    Status coding:
    0 -> 0-75% full
    1 -> 76-90% full
    2 -> 91-100% full
    3 -> Under maintenance
    4 -> Under maintenance, over capacity
    """
    # Treat data
    dataSplit = [
        [data for data in sensor.split("$#$")] for sensor in raw.split("$#$$@$")
    ]
    data = []
    for record in dataSplit:
        # Convert data to dict, removing duplicates
        if record not in data and len(record) == 7:
            data.append(
                {
                    "sensor-id": record[0],
                    "sensor-name": record[1],
                    "latitude": float(record[3]),
                    "longitude": float(record[2]),
                    "water-level": float(record[4]),
                    "status": float(record[5]),
                    "timestamp": parseTimestamp(record[6]),
                }
            )
    df = pd.DataFrame(data)
    return df


raw = getFlooding(datetime(year=2023, month=7, day=20))
data = parseFlooding(raw)
data.to_csv("data.csv", index=False)
