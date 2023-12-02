# Functions for pulling data from government APIs
try:
    from .myutils import (
        normalizeWeather,
        normalizeStations,
    )
except:
    from myutils import (
        normalizeWeather,
        normalizeStations,
    )
from datetime import datetime
from colorama import Fore, Back, Style
from time import time
import requests, pandas as pd


def getWeather(type: str, date: datetime) -> pd.DataFrame:
    """
    Gets raw data using the NEA's realtime weather API for the given date, of a specific type.\n

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
            if not "value" in reading:
                reading["value"] = None
            reading[type] = reading.pop("value")

    # Merge required data
    stations = normalizeStations(response["metadata"]["stations"])
    readings = normalizeWeather(timedReadings)
    readings = readings.merge(right=stations, on="station-id", how="left")
    readings = readings[
        ["timestamp", "station-id", "station-name", "latitude", "longitude", type]
    ]

    # Log and return data
    print(f"{round(time()-weatherTimer,2)}s")
    return readings


def getWeatherRange(
    date: datetime, endDate: datetime = None, interval: int = 5
) -> pd.DataFrame:
    """
    Gets detailed weather data over a range of dates using the NEA's realtime weather API.

    Weather Data
    ----------
    `air-temperature`: deg C\n
    `relative-humidity`: %\n
    `rainfall`: mm\n
    `wind-direction`: degrees\n
    `wind-speed`: knots\n

    Parameters
    ----------
    `date`: Date on which data should begin. To fetch data for a single date, do not pass in endDate\n
    `endDate`: Optional date on which data should end\n
    `interval`: Interval in minutes between each reading. Mean of measurements within each interval is taken as measurement for that interval.
    """

    # Log time
    allTimer = time()
    # Get list of dates to fetch
    if endDate == None:
        endDate = date
    dates = list(pd.date_range(date, endDate))
    weatherDf = pd.DataFrame()

    # Get data over date range
    for date in dates:
        # Begin logging
        dateTimer = time()
        print(
            Fore.BLACK + Back.GREEN + "[FETCH]" + Style.RESET_ALL,
            f"Weather for {date.strftime('%d/%m/%Y')}",
        )

        # Get all weather/station data
        weatherTypes = [
            "rainfall",
            "air-temperature",
            "relative-humidity",
            "wind-direction",
            "wind-speed",
        ]
        weatherData = []
        for type in weatherTypes:
            weatherData.append(getWeather(type, date))

        # Merge weather data for a particular day and add to weatherDf
        dateDf = weatherData[0]
        for i in range(1, len(weatherData)):
            currentType = weatherData[i].columns.tolist()[-1]
            dateDf = dateDf.merge(
                weatherData[i][["timestamp", "station-id", currentType]],
                how="outer",
                on=["timestamp", "station-id"],
            )
        weatherDf = pd.concat(
            [weatherDf, dateDf.reset_index(drop=True)], ignore_index=True
        )
        print(
            Fore.BLACK + Back.GREEN + "[FETCH]" + Style.RESET_ALL,
            f"Completed in {round(time()-dateTimer, 2)}s\n",
        )

    # Clean and neaten data
    weatherDf = (
        weatherDf.infer_objects()
        .sort_values(by=["station-id", "timestamp"])
        .reset_index(drop=True)
    )
    weatherDf = weatherDf.reindex(
        columns=[
            "timestamp",
            "station-id",
            "station-name",
            *weatherDf.columns.tolist()[3:],
        ]
    )
    # Compress to defined interval
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

    # Log ending and return result
    print(
        Fore.BLACK
        + Back.GREEN
        + f"Data fetched in {round(time()-allTimer,2)}s"
        + Style.RESET_ALL
    )
    return weatherDf
