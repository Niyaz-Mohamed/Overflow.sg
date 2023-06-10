# General utilities for pulling data
import requests, json, os
import pandas as pd
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


def fetchWeather(type: str, date: datetime):
    """
    NOTE: Using fetchWeatherRange() is preferred.\n
    Fetches raw data using the NEA's reatime weather API for the given date, of a specific type.

    Parameters
    ----------
    `startDate`: Date on which data should begin\n
    `endDate`: Date on which data should end, defaults to same day as startDate\n
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
    Normalizes and cleans weather readings, converting it to a flattened dataframe.

    Parameters
    ----------
    `data`: Raw weather data in dictionary format\n
    """
    # Flatten json
    df = pd.json_normalize(data, record_path="readings", meta=["timestamp"])
    # Clean up by reordering and sorting columns
    cols = df.columns.to_list()
    cols = cols[-1:] + cols[:-1]
    df = df[cols].sort_values(by=["timestamp", "station_id"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%dT%H:%M:%S%z")
    return df.reset_index(drop=True)


def normalizeStations(data: dict) -> pd.DataFrame:
    """
    Normalizes and cleans station data, converting it to a flattened dataframe.

    Parameters
    ----------
    `data`: Raw station data in dictionary format\n
    """
    # Flatten json and clean df
    df = pd.json_normalize(data)
    df = df.drop(columns="device_id").rename(
        columns={
            "id": "station_id",
            "name": "station_name",
            "location.latitude": "latitude",
            "location.longitude": "longitude",
        }
    )
    return df


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


def concatStations(data: list[pd.DataFrame], how: str = "left"):
    """Merges different station dataframes together.

    Parameters
    ----------
    `data`: List of dataframes to merge. First dataframe is taken as the left/right dataframe\n
    `how`: Merge type (inner, outer, left, right)\n
    """
    df = data[0]
    for i in range(1, len(data)):
        df = pd.concat([df, data[i]], ignore_index=True).drop_duplicates()
    return df.reset_index(drop=True)


def fetchWeatherRange(startDate: datetime, endDate: datetime = None):
    """Fetches detailed weather data over a range of dates using the NEA's reatime weather API.

    Parameters
    ----------
    `startDate`: Date on which data should begin\n
    `endDate`: Date on which data should end, defaults to same day as startDate\n
    """
    if endDate == None:
        endDate = startDate
    dates = list(pd.date_range(startDate, endDate))
    weatherDf = pd.DataFrame()

    for date in dates:
        # Fetch all weather/station data
        weatherTypes = [
            "air-temperature",
            "relative-humidity",
            "rainfall",
            "wind-direction",
            "wind-speed",
        ]
        weatherData = []
        stationData = []
        for type in weatherTypes:
            apiData = fetchWeather(type, date)
            weatherData.append(normalizeWeather(apiData["readings"]))
            stationData.append(normalizeStations(apiData["stations"]))

        # Merge weather data and station data
        mergedStations = concatStations(stationData)
        mergedWeather = mergeWeather(weatherData).merge(
            mergedStations, how="left", on="station_id"
        )
        weatherDf = pd.concat([weatherDf, mergedWeather], ignore_index=True)

    # Reorder weatherDf
    cols = weatherDf.columns.tolist()
    cols = cols[:2] + cols[-3:] + cols[2:-3]
    print(cols)
    weatherDf = weatherDf[cols]
    return weatherDf.reset_index(drop=True)
