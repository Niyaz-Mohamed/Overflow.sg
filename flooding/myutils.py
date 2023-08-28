from datetime import datetime
import pandas as pd


def parseFlooding(raw: str):
    """
    Parses raw flooding data received from the API into a dataframe.
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
                    "timestamp": parseTimestamp(record[6]),
                    "sensor-id": record[0],
                    "sensor-name": record[1],
                    "latitude": float(record[3]),
                    "longitude": float(record[2]),
                    "water-level": float(record[4]),
                    "status": int(record[5]),
                }
            )
    df = pd.DataFrame(data)
    return df.sort_values(by=["timestamp", "sensor-id"]).reset_index(drop=True)


def parseTimestamp(timestamp: str):
    """
    Parses flood observation timestamps, accounting for possible errors/format issues.
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
