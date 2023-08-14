# Functions for pulling data from government APIs
from myutils import normalizeWeather, normalizeStations, mergeWeather, concatStations
from datetime import datetime
from colorama import Back, Style
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
    print(
        Back.WHITE + "[FETCH]" + Style.RESET_ALL,
        f"{type}",
    )
    # Fetch data
    endpoint = "https://api.data.gov.sg/v1/environment/" + type
    params = {"date": date.strftime("%Y-%m-%d")}
    response = requests.get(endpoint, params=params).json()

    # Change keys for readings to match type of reading, adding units
    timedReadings = response["items"]
    for minute in range(len(timedReadings)):
        for reading in timedReadings[minute]["readings"]:
            reading[type] = reading.pop("value")

    return {
        "raw-data": response,
        "stations": normalizeStations(response["metadata"]["stations"]),
        "readings": normalizeWeather(timedReadings),
        "type": type,
        "api-status": response["api_info"]["status"],
    }


def fetchWeatherRange(startDate: datetime, endDate: datetime = None):
    """
    Fetches detailed weather data over a range of dates using the NEA's realtime weather API.

    Parameters
    ----------
    `startDate`: Date on which data should begin\n
    `endDate`: Date on which data should end, defaults to same day as startDate\n
    """

    # Get list of dates to fetch
    if endDate == None:
        endDate = startDate
    dates = list(pd.date_range(startDate, endDate))
    weatherDf = pd.DataFrame()

    for date in dates:
        # Log details
        print(
            Back.GREEN + "[DATE]" + Style.RESET_ALL,
            f"Weather data for {date.strftime('%d/%m/%Y')}",
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

    # Reorder columns
    cols = weatherDf.columns.tolist()
    cols = cols[:2] + cols[-3:] + cols[2:-3]
    weatherDf = weatherDf[cols]

    # Typecast/clean columns
    weatherDf = weatherDf.infer_objects()
    unitMap = {
        "air-temperature": "oC",
        "relative-humidity": "%",
        "rainfall": "mm",
        "wind-direction": "deg",
        "wind-speed": "knots",
    }
    for reading in unitMap:
        unit = unitMap[reading]
        weatherDf = weatherDf.rename(columns={reading: f"{reading} ({unit})"})
    return weatherDf.sort_values(by=["station-id", "timestamp"]).reset_index(drop=True)
