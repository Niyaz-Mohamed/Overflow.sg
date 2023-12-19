from weather import getWeatherRange, weatherAt
from flooding import fetchFromDatabase
import os, pandas as pd
from datetime import datetime, timedelta
from geopy.distance import geodesic
from colorama import Back, Style
from time import time
from typing import Union, Iterable


def log(color: str, foreText: str = "", bodyText: str = "", end: str = ""):
    """
    Convenience function for logging, accepts a color and creates a printed log.

    NOTE: color should be a colorama background color
    """
    print(color + foreText + Style.RESET_ALL + " " + bodyText)


def getAllData():
    """
    Fetches all flooding and relevant weather data and memoizes
    it in a data folder in the same directory as the current file.\n\n
    Returns a floodDf and weatherDf
    """
    # Check if flooding data exists, generate it if otherwise
    floodDataPath = os.path.join(__file__, "../data/floodData.csv")
    if not os.path.isfile(floodDataPath):
        floodDf = (
            # Filter unneeded documents
            pd.DataFrame(
                fetchFromDatabase(
                    query={"status": {"$in": [0, 1, 2]}},
                )
            )
            # Clean and save data
            .drop(
                columns=["_id", "timestamp (data fetched)", "water-level", "max-level"]
            ).sort_values(by="timestamp", ascending=True)
        )
        floodDf.to_csv(floodDataPath, index=False)
    # Load memoized flooding data
    floodDf = pd.read_csv(floodDataPath)

    # Check if weather data is suitable for flood data
    weatherDataPath = os.path.join(__file__, "../data/weatherData.csv")
    floodMin = datetime.fromisoformat(floodDf["timestamp"].min()).date()
    floodMax = datetime.fromisoformat(floodDf["timestamp"].max()).date()

    # Generate data if i1t doesn't exist
    if not os.path.isfile(weatherDataPath):
        weatherDf = getWeatherRange(floodMin - timedelta(days=2), floodMax)
        weatherDf.to_csv(weatherDataPath, index=False)
    # Define conditions on which to add more dates
    else:
        weatherDf = pd.read_csv(weatherDataPath)
        weatherMin = datetime.fromisoformat(weatherDf["timestamp"].min()[:-6]).date()
        weatherMax = datetime.fromisoformat(weatherDf["timestamp"].max()[:-6]).date()
        prependDates = weatherMin + timedelta(days=2) > floodMin
        appendDates = weatherMax < floodMax

        # Prepend missing early dates
        if prependDates:
            prependDf = getWeatherRange(floodMin - timedelta(days=2), weatherMin)
            weatherDf = pd.concat([prependDf, weatherDf])

        # Append missing end dates
        if appendDates:
            print(floodMax)
            appendDf = getWeatherRange(weatherMax + timedelta(days=1), floodMax)
            weatherDf = pd.concat([weatherDf, appendDf])

        # Update csv file
        if appendDates or prependDates:
            weatherDf.drop_duplicates().to_csv(weatherDataPath, index=False)

    # Fix typings for dataframes
    weatherDf["timestamp"] = weatherDf["timestamp"].str[:-6]
    weatherDf["timestamp"] = pd.to_datetime(weatherDf["timestamp"])
    floodDf["timestamp"] = pd.to_datetime(floodDf["timestamp"])
    # Cut out all rows with nils
    weatherDf.dropna(inplace=True)
    return floodDf, weatherDf


def calculateClosestStation(
    floodDf: pd.DataFrame, weatherDf: pd.DataFrame
) -> pd.DataFrame:
    """
    Injects information about the closest weather station into floodDf\n
    and memoizes it in a data folder in the same directory as the current file.

    NOTE: Always delete
    """

    # Check if file exists
    memoDataPath = os.path.join(__file__, "../data/floodDataWithStations.csv")
    if os.path.isfile(memoDataPath):
        # Check if file is valid (number of rows equal)
        memoDf = pd.read_csv(memoDataPath)
        if memoDf.shape[0] == floodDf.shape[0]:
            return memoDf

    # Quickly match sensors to stations
    sensors = floodDf.drop_duplicates(subset="sensor-id", keep="first")[
        ["sensor-id", "sensor-name", "latitude", "longitude"]
    ]
    stations = weatherDf.drop_duplicates(subset="station-id", keep="first")[
        ["timestamp", "station-id", "station-name", "latitude", "longitude"]
    ]

    # Calculate distance by comparing rows in dataframes
    def calcDistance(row):
        sensorCoords = (row["latitude"], row["longitude"])
        distances = [
            geodesic(
                sensorCoords, (station["latitude"], station["longitude"])
            ).kilometers
            for _, station in stations.iterrows()
        ]
        closestStation = stations.iloc[distances.index(min(distances))]
        return pd.Series(
            {
                "station-distance": min(distances),
                "station-id": closestStation["station-id"],
                "station-name": closestStation["station-name"],
                "station-latitude": closestStation["latitude"],
                "station-longitude": closestStation["longitude"],
                "station-first-timestamp": closestStation["timestamp"],
            }
        )

    # Apply distance calculation function and filter columns (only keep sensor id and station details)
    sensors = pd.concat([sensors, sensors.apply(calcDistance, axis=1)], axis=1)
    col = sensors.columns.tolist()
    sensors = sensors[col[:1] + col[4:]]
    # Merge stations onto flooding data
    floodDf = floodDf.merge(right=sensors, on="sensor-id")

    # TODO: Account for stations having gaps in their recording periods
    # Function handling error values (timestamp of flooding before first timestamp of station)
    # through iteration of flooding data
    def calcDistanceManual(row):
        sensorCoords = (row["latitude"], row["longitude"])
        availStations = stations[stations["timestamp"] <= row["timestamp"]]
        distances = [
            geodesic(
                sensorCoords, (station["latitude"], station["longitude"])
            ).kilometers
            for _, station in availStations.iterrows()
        ]
        closestStation = availStations.iloc[distances.index(min(distances))]
        return pd.Series(
            {
                "station-distance": min(distances),
                "station-id": closestStation["station-id"],
                "station-name": closestStation["station-name"],
                "station-latitude": closestStation["latitude"],
                "station-longitude": closestStation["longitude"],
                "station-first-timestamp": closestStation["timestamp"],
            }
        )

    # Get all rows with error values and fix errors
    errorDf = floodDf[floodDf["timestamp"] < floodDf["station-first-timestamp"]]
    cols = errorDf.columns.tolist()
    errorDf = errorDf[cols[:-6]]
    errorDf = pd.concat([errorDf, errorDf.apply(calcDistanceManual, axis=1)], axis=1)
    # Update the original dataset with the corrected values and clean df
    floodDf.update(errorDf)
    floodDf = (
        floodDf.rename(
            {"latitude": "sensor-latitude", "longitude": "sensor-longitude"}, axis=1
        )
        .sort_values(by="timestamp", ascending=True)
        .reset_index(drop=True)
    )
    cols = floodDf.columns.tolist()
    floodDf = floodDf[cols[:5] + cols[8:-1] + [cols[7], cols[5]]]

    # Save and memoize the dataframe
    floodDf.to_csv(memoDataPath, index=False)
    floodDf = pd.read_csv(memoDataPath)
    return floodDf


def injectWeatherData(
    floodDf: pd.DataFrame,
    weatherDf: pd.DataFrame,
    predictionTime: int,
    intervalSize: int,
    numReadings: int,
    readingSize: int,
) -> pd.DataFrame:
    """
    Injects approppriate weather data into floodDf based on some parameters

    Parameters
    -----------
    `predictionTime`: Prediction time (in hours) that the AI produced will have\n
    `intervalSize`: Length of interval (in hours) between each moment in time where weather is measured\n
    `numreadings`: Number of moments in time where weather is measured (prior to prediction time)\n
    `readingSize`: Size of period around reading time (in minutes), taken to be reading for that time\n
    """

    timeShifts = [predictionTime + n * intervalSize for n in range(numReadings)]

    # Function to add weather for a particular time shift
    # Time complexity n based on numReadings
    def addWeatherSet(row):
        # Log progress occationally
        if (row.name + 1) % 2000 == 0:
            log(
                Back.WHITE,
                "[UPDATE]",
                f"Completed {row.name+1}/{floodDf.shape[0]} rows",
            )

        weatherSet = [
            weatherAt(
                datetime.fromisoformat(row["timestamp"].replace(" ", "T"))
                - timedelta(hours=timeShift),
                weatherDf,
                interval=readingSize,
                stationId=row["station-id"],
            )
            for timeShift in timeShifts
        ]

        # Construct result
        resultWeather = [
            pd.Series(
                {
                    f"rainfall-{timeShifts[i]}h-prior": weatherSet[i]["rainfall"],
                    f"air-temperature-{timeShifts[i]}h-prior": weatherSet[i][
                        "air-temperature"
                    ],
                    f"relative-humidity-{timeShifts[i]}h-prior": weatherSet[i][
                        "relative-humidity"
                    ],
                    f"wind-direction-{timeShifts[i]}h-prior": weatherSet[i][
                        "wind-direction"
                    ],
                    f"wind-speed-{timeShifts[i]}h-prior": weatherSet[i]["wind-speed"],
                }
            )
            for i in range(len(weatherSet))
        ]
        resultWeather = pd.concat(resultWeather)
        return resultWeather

    # Apply the function to create multiple new columns
    # Time complexity n based on number of rows
    newColumns = floodDf.apply(addWeatherSet, axis=1)
    # Merge the original DataFrame with the new columns
    floodDf = pd.concat([floodDf, newColumns], axis=1)
    # Push the '% full' column to the right
    cols = floodDf.columns.tolist()
    cols.append(cols[9])
    del cols[9]
    floodDf = floodDf[cols]
    return floodDf


def constructDataset(
    predictionTime: int,
    intervalSize: int,
    numReadings: int,
    readingSize: int,
    restrictDistance: int = None,
    restrictRows: int = None,
    restrictDate: Union[str, Iterable] = None,
    saveName: str = "trainingData.csv",
) -> pd.DataFrame:
    """
    Constructs a training dataset based on parameters given.\n\n
    Returns a training dataset which it will also memoize in a data folder.

    Parameters
    -----------
    `predictionTime`: Prediction time (in hours) that the AI produced will have\n
    `intervalSize`: Length of interval (in hours) between each moment in time where weather is measured\n
    `numreadings`: Number of moments in time where weather is measured (prior to prediction time)\n
    `readingSize`: Size of period around reading time (in minutes), taken to be reading for that time\n
    `restrictDistance`: Only accept rows where station-to-sensor distance is lower than this [optional]\n
    `restrictRows`: Restrict number of rows of filtered dataframe to inject weather into\n
    `restrictDate`: 1 or 2 date values to filter dates from start to end time, formatted as '2023-01-30'\n
    `saveName`: Filename for saving the file, 'trainingData.csv' by default
    """

    # Fetch base datasets
    log(Back.CYAN, "[TASK]", "Fetching and saving base weather and flooding datasets")
    floodDf, weatherDf = getAllData()

    # Restrict date range
    if restrictDate:
        # Calculate approppriate start and end times
        try:
            startDate = pd.to_datetime(restrictDate[0] + "T00:00:00")
            endDate = pd.to_datetime(restrictDate[1] + "T23:59:59")
            log(
                Back.CYAN,
                "[TASK]",
                f"Restricting dates in flood dataset to between {restrictDate[0]} and {restrictDate[1]}",
            )
        except:
            log(
                Back.CYAN,
                "[TASK]",
                f"Restricting dates in flood dataset to {restrictDate}",
            )
            startDate = pd.to_datetime(restrictDate + "T00:00:00")
            endDate = pd.to_datetime(restrictDate + "T23:59:59")

        # Apply filters
        floodDf = floodDf[
            (floodDf["timestamp"] >= startDate) & (floodDf["timestamp"] <= endDate)
        ].reset_index(drop=True)

    # Match water level sensors to closest weather station
    log(Back.CYAN, "[TASK]", "Matching weather stations to flood sensors")
    floodDf = calculateClosestStation(floodDf, weatherDf)

    # Restrict rows and distance
    if restrictDistance:
        log(
            Back.CYAN,
            "[TASK]",
            f"Restricting station-distance in flood dataset to below {restrictDistance}km",
        )
        floodDf = floodDf[floodDf["station-distance"] < restrictDistance]
        floodDf.reset_index(drop=True, inplace=True)
    if restrictRows:
        log(
            Back.CYAN,
            "[TASK]",
            f"Restricting row count in flood dataset to {restrictRows}",
        )
        floodDf = floodDf.head(restrictRows)

    # Check for existing dataset (only after fetching and saving original datasets)
    savePath = os.path.join(__file__, f"../data/{saveName}")
    if os.path.isfile(savePath):
        log(
            Back.GREEN,
            "Existing complete dataset found, ignoring row restriction",
            "\n",
        )
        return pd.read_csv(savePath)

    # Inject weather data into flooding data based on factors (EXPENSIVE)
    log(Back.CYAN, "[TASK]", "Injecting weather data into flooding dataset")
    expectedTime = (270 * numReadings * floodDf.shape[0]) / (12000 * 3)
    log(
        Back.RED,
        "[NOTE]",
        f"Expected time required is {round(expectedTime)}s ({floodDf.shape[0]} rows, {numReadings} readings)",
    )
    startTime = time()
    floodDf = injectWeatherData(
        floodDf,
        weatherDf,
        predictionTime,
        intervalSize,
        numReadings,
        readingSize,
    )
    log(Back.GREEN, f"Completed in {round(time()-startTime,2)}s", "\n")

    # Save dataset to csv
    savePath = os.path.join(__file__, f"../data/{saveName}")
    floodDf = floodDf.dropna()
    floodDf.to_csv(savePath, index=False)
    return floodDf
