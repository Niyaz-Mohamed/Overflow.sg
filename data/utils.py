# General utilities for pulling data
import requests, json, os
import pandas as pd
import numpy as np
from datetime import datetime


def genJSON(data: dict, filename: str = "data"):
    """
    Creates a [filename].json file using the data passed in.
    File is created in the same directory as the script.
    """
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__))
    )
    file = open(os.path.join(__location__, f"{filename}.json"), "w")
    file.write(json.dumps(data))
    file.close()


def fetchWeatherData(type, date: datetime):
    """
    Fetches raw data using the NEA's reatime weather API at 1h intervals.

    type: Can be air-temperature, relative-humidity, rainfall, wind-direction or wind-speed
    date: Date from which data is taken
    """

    # Fetch data
    endpoint = "https://api.data.gov.sg/v1/environment/" + type
    params = {"date": date.strftime("%Y-%m-%d")}
    response = requests.get(endpoint, params=params).json()

    # Separate and filter data to 1h granularity
    stations = response["metadata"]["stations"]

    def filterReading(reading):
        timestamp = datetime.strptime(reading["timestamp"], "%Y-%m-%dT%H:%M:%S%z")
        return timestamp.minute == 1

    timedReadings = list(
        filter(
            filterReading,
            response["items"],
        )
    )

    # Change 'value' keys to match type
    for i in range(len(timedReadings)):
        for reading in timedReadings[i]["readings"]:
            reading[str(type)] = reading.pop("value")

    return {
        "stations": stations,
        "readings": timedReadings,
        "type": type,
        "api-status": response["api_info"]["status"],
    }


def normalizeWeather(data: dict) -> pd.DataFrame:
    """
    Normalizes and cleans weather readings in dictionary format, generating a flattened dataframe.
    """
    # Flatten json
    df = pd.json_normalize(readings, record_path="readings", meta=["timestamp"])
    # Clean up by reordering and sorting columns
    cols = df.columns.to_list()
    cols = cols[-1:] + cols[:-1]
    df = df[cols].sort_values(by=["timestamp", "station_id"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%dT%H:%M:%S%z")
    return df.reset_index(drop=True)


#! Test code ONLY
date = datetime(year=2023, month=6, day=5)
readings = fetchWeatherData("air-temperature", date)["readings"]
df = normalizeWeather(readings)
print(df.dtypes)
print("---------------------------------------")
print(df.head())
