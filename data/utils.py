# General utilities for pulling data
import requests, json, os
import pandas as pd
from datetime import datetime, timedelta


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


# todo: Change behavior to aggregate data
def fetchWeather(type: str, date: datetime, interval: int = 60):
    """
    NOTE: Using fetchWeatherRange() is preferred
    Fetches raw data using the NEA's reatime weather API at the specified intervals, and aggregates that data over the interval.
    """

    # Fetch data
    endpoint = "https://api.data.gov.sg/v1/environment/" + type
    params = {"date": date.strftime("%Y-%m-%d")}
    response = requests.get(endpoint, params=params).json()

    # Separate and filter data to 1h granularity
    stations = response["metadata"]["stations"]

    def filterReading(reading):
        timestamp = datetime.strptime(reading["timestamp"], "%Y-%m-%dT%H:%M:%S%z")
        return timestamp.minute == 0

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
    Normalizes and cleans weather readings in dictionary format to a flattened dataframe.
    """
    # Flatten json
    df = pd.json_normalize(data, record_path="readings", meta=["timestamp"])
    # Clean up by reordering and sorting columns
    cols = df.columns.to_list()
    cols = cols[-1:] + cols[:-1]
    df = df[cols].sort_values(by=["timestamp", "station_id"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%dT%H:%M:%S%z")
    return df.reset_index(drop=True)


def mergeWeather(
    data: list[pd.DataFrame], how: str = "left", on: list = ["timestamp", "station_id"]
):
    """Merges different weather dataframes together.

    Parameters
    ----------
    `data`: List of dataframes to merge. First dataframe is taken as the left/right dataframe\n
    `how`: Merge type (inner, outer, left, right)\n
    `on`: Columns to use as keys when merging
    """
    df = data[0]
    for i in range(1, len(data)):
        df = df.merge(data[i], how=how, on=on)
    return df.reset_index(drop=True)


def fetchWeatherRange(startDate: datetime, endDate: datetime, interval: int = 60):
    """Fetches and processes weather over a date range using the NEA's reatime weather API at the specified interval.

    Parameters
    ----------
    `type`: Can be air-temperature, relative-humidity, rainfall, wind-direction or wind-speed\n
    `date`: Date from which data is taken\n
    `interval`: Intervals, in minutes, between each reading
    """
    dates = list(pd.date_range(startDate, endDate))

    for date in dates:
        # Fetch all weather data and merge
        weatherTypes = [
            "air-temperature",
            "relative-humidity",
            "rainfall",
            "wind-direction",
            "wind-speed",
        ]
        temp, humidity, rainfall, windDirection, windSpeed = [
            normalizeWeather(fetchWeather(type, date, interval))
            for type in weatherTypes
        ]
        mergedWeather = mergeWeather(
            [temp, humidity, rainfall, windDirection, windSpeed]
        )

        # Combine data from multiple dates
        if date == date[0]:
            df = mergedWeather
        else:
            dfAdded = mergedWeather
            df = pd.concat([df, dfAdded], ignore_index=True)

        return df.reset_index(drop=True)


#! Test code ONLY
startDate = datetime(year=2023, month=6, day=6)
endDate = datetime(year=2023, month=6, day=6)
df = fetchWeatherRange(startDate, endDate)
print(df.head())
