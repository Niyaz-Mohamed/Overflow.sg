# Functions for pulling data from government APIs
from myutils import normalizeWeather, normalizeStations, mergeWeather, concatStations
from datetime import datetime
from colorama import Fore, Back, Style
from time import time
import requests, pandas as pd


def fetchWeather(type: str, date: datetime):
    """
    Fetches raw data using the NEA's realtime weather API for the given date, of a specific type. Data is fetched at 5 minute intervals.
    NOTE: Measurement types used for each reading are:\n\t
        `air-temperature`: DBT 1M F\n\t
        `relative-humidity`: RH 1M F\n\t
        `rainfall`: TB1 Rainfall 5 Minute Total F\n\t
        `wind-direction`: Wind Dir AVG (S) 10M M1M\n\t
        `wind-speed`: Wind Speed AVG(S)10M M1M\n\t

    Parameters
    ----------
    `type`: Type of data to pull (air-temperature, relative-humidity, rainfall, wind-direction, wind-speed)\n
    `endDate`: Date on which data should end, defaults to same date as startDate
    """
    # Log details
    print(Fore.BLACK + Back.WHITE + "[GET]" + Style.RESET_ALL, f"{type}", end=": ")
    weatherTimer = time()
    # Fetch data
    endpoint = "https://api.data.gov.sg/v1/environment/" + type
    params = {"date": date.strftime("%Y-%m-%d")}
    response = requests.get(endpoint, params=params).json()

    # Change keys for readings to match type of reading, adding units
    timedReadings = response["items"]
    for minute in range(len(timedReadings)):
        for reading in timedReadings[minute]["readings"]:
            reading[type] = reading.pop("value")

    # Log and return data
    print(f"{round(time()-weatherTimer,2)}s")
    return {
        "raw-data": response,
        "stations": normalizeStations(response["metadata"]["stations"]),
        "readings": normalizeWeather(timedReadings),
        "type": type,
        "api-status": response["api_info"]["status"],
    }


def fetchWeatherRange(startDate: datetime, endDate: datetime = None, interval: int = 5):
    """
    Fetches detailed weather data over a range of dates using the NEA's realtime weather API.

    Weather Data
    ----------
    `air-temperature`: deg C\n
    `relative-humidity`: %\n
    `rainfall`: mm\n
    `wind-direction`: degrees\n
    `wind-speed`: knots\n

    Parameters
    ----------
    `startDate`: Date on which data should begin\n
    `endDate`: Date on which data should end, defaults to same day as startDate\n
    `interval`: Interval in minutes between each reading. Mean of measurements within each interval is taken as measurement for that interval.
    """

    # Log time
    allTimer = time()
    # Get list of dates to fetch
    if endDate == None:
        endDate = startDate
    dates = list(pd.date_range(startDate, endDate))
    weatherDf = pd.DataFrame()

    for date in dates:
        dateTimer = time()
        # Log details
        print(
            Fore.BLACK + Back.GREEN + "[FETCH]" + Style.RESET_ALL,
            f"Weather for {date.strftime('%d/%m/%Y')}",
        )
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
            weatherData.append(apiData["readings"])
            stationData.append(apiData["stations"])

        # Merge weather data and station data
        mergedStations = concatStations(stationData)
        mergedWeather = mergeWeather(weatherData).merge(
            mergedStations, how="left", on="station-id"
        )
        weatherDf = pd.concat([weatherDf, mergedWeather], ignore_index=True)
        print(
            Fore.BLACK + Back.GREEN + "[FETCH]" + Style.RESET_ALL,
            f"Completed in {round(time()-dateTimer, 2)}s\n",
        )

    # Typecast, clean, and compress columns to interval
    weatherDf = (
        weatherDf.infer_objects()
        .sort_values(by=["station-id", "timestamp"])
        .reset_index(drop=True)
    )
    weatherDf = weatherDf.groupby(
        [
            "station-id",
            "station-name",
            pd.Grouper(
                freq=f"{interval}min",
                key="timestamp",
                origin="start",
                label="right",
                closed="right",
                offset="-1min",
            ),
        ],
        as_index=False,
    ).mean()

    # Reorder columns
    cols = weatherDf.columns.tolist()
    cols = [cols[2]] + cols[:2] + cols[-2:] + [cols[5]] + cols[3:5] + cols[6:-2]
    weatherDf = weatherDf[cols]
    print(
        Fore.BLACK
        + Back.GREEN
        + f"Data fetched in {round(time()-allTimer,2)}s"
        + Style.RESET_ALL
    )
    return weatherDf
