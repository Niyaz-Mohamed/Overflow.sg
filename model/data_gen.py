from weather import getWeatherRange, weatherAt
from flooding import fetchFromDatabase
import os, pandas as pd
from datetime import datetime, timedelta
from geopy.distance import geodesic


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
    floodMin = datetime.fromisoformat(floodDf["timestamp"].min())
    floodMax = datetime.fromisoformat(floodDf["timestamp"].max())

    # Generate data if i1t doesn't exist
    if not os.path.isfile(weatherDataPath):
        weatherDf = getWeatherRange(floodMin - timedelta(days=2), floodMax)
        weatherDf.to_csv(weatherDataPath, index=False)
    # Define conditions on which to add more dates
    else:
        weatherDf = pd.read_csv(weatherDataPath)
        weatherMin = datetime.fromisoformat(weatherDf["timestamp"].min()[:-6])
        weatherMax = datetime.fromisoformat(weatherDf["timestamp"].max()[:-6])
        prependDates = weatherMin + timedelta(days=2) > floodMin
        appendDates = weatherMax < floodMax

        # Prepend missing early dates
        if prependDates:
            prependDf = getWeatherRange(floodMin - timedelta(days=2), weatherMin)
            weatherDf = pd.concat([prependDf, weatherDf])

        # Append missing end dates
        if appendDates:
            appendDf = getWeatherRange(weatherMax, floodMax)
            weatherDf = pd.concat([weatherDf, appendDf])

        # Update csv file
        if appendDates or prependDates:
            weatherDf.drop_duplicates().to_csv(weatherDataPath, index=False)

    # Infer types for weatherDf
    weatherDf["timestamp"] = weatherDf["timestamp"].str[:-6]
    weatherDf["timestamp"] = pd.to_datetime(weatherDf["timestamp"])
    return floodDf, weatherDf


def calculateClosestStation(
    floodDf: pd.DataFrame, weatherDf: pd.DataFrame
) -> pd.DataFrame:
    """
    Injects information about the closest weather station into floodDf.
    """
    # TODO: Iterate through floodDf and join in closest station with data for the given time
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
                "station-longitude": closestStation["latitude"],
            }
        )

    # Apply distance calculation function and filter columns
    sensors = pd.concat([sensors, sensors.apply(calcDistance, axis=1)], axis=1)[
        [
            "sensor-id",
            "station-id",
            "station-name",
            "station-latitude",
            "station-longitude",
            "station-distance",
        ]
    ]

    # Merge stations onto flooding data
    floodDf = (
        floodDf.merge(right=sensors, on="sensor-id")
        .sort_values(by="timestamp", ascending=True)
        .rename(
            {"latitude": "sensor-latitude", "longitude": "sensor-longitude"}, axis=1
        )
        .reindex(
            columns=[
                "timestamp",
                "sensor-id",
                "sensor-name",
                "sensor-latitude",
                "sensor-longitude",
                "station-id",
                "station-name",
                "station-latitude",
                "station-longitude",
                "station-distance",
                "% full",
            ]
        )
        .reset_index(drop=True)
    )
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

    timeShifts = [-(predictionTime + n * intervalSize) for n in range(numReadings)]

    for i in range(len(timeShifts)):
        timeShift = timeShifts[i]

        # Function to add weather for a particular time shift
        def addWeatherSet(row):
            weatherSet = weatherAt(
                datetime.fromisoformat(row["timestamp"].replace(" ", "T")),
                weatherDf,
                interval=readingSize,
                stationId=row["station-id"],
            )
            return pd.Series(
                {
                    f"rainfall{timeShift}h-prior": weatherSet["rainfall"],
                    f"air-temperature{timeShift}h-prior": weatherSet["air-temperature"],
                    f"relative-humidity{timeShift}h-prior": weatherSet[
                        "relative-humidity"
                    ],
                    f"wind-direction{timeShift}h-prior": weatherSet["wind-direction"],
                    f"wind-speed{timeShift}h-prior": weatherSet["wind-speed"],
                }
            )

        # Apply the function to create multiple new columns
        newColumns = floodDf.apply(addWeatherSet, axis=1)
        # Merge the original DataFrame with the new columns
        floodDf = pd.concat([floodDf, newColumns], axis=1)

    return floodDf


# Get data and find nearest station
floodDf, weatherDf = getAllData()
floodDf = calculateClosestStation(floodDf, weatherDf)
# Inject weather data into flooding data based on factors
floodDf = injectWeatherData(
    floodDf,
    weatherDf,
    predictionTime=1,
    intervalSize=0.5,
    numReadings=3,
    readingSize=10,
)
print(floodDf)
