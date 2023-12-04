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
                "station-longitude": closestStation["latitude"],
                "station-first-timestamp": closestStation["timestamp"],
            }
        )

    # Apply distance calculation function and filter columns (only keep sensor id and station details)
    sensors = pd.concat([sensors, sensors.apply(calcDistance, axis=1)], axis=1)
    col = sensors.columns.tolist()
    sensors = sensors[col[:1] + col[4:]]
    # Merge stations onto flooding data
    floodDf = floodDf.merge(right=sensors, on="sensor-id")

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
                "station-longitude": closestStation["latitude"],
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
    floodDf = floodDf[cols[:5] + cols[8:-1] + cols[5:6]]

    # Save and memoize the dataframe
    floodDf.to_csv(memoDataPath, index=False)
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

    # Function to add weather for a particular time shift
    def addWeatherSet(row):
        weatherSet = [
            weatherAt(
                datetime.fromisoformat(row["timestamp"].replace(" ", "T"))
                + timedelta(hours=timeShift),
                weatherDf,
                interval=readingSize,
                stationId=row["station-id"],
            )
            for timeShift in timeShifts
        ]
        # Construct result
        res = [
            pd.Series(
                {
                    f"rainfall{timeShifts[i]}h-prior": weatherSet[i]["rainfall"],
                    f"air-temperature{timeShifts[i]}h-prior": weatherSet[i][
                        "air-temperature"
                    ],
                    f"relative-humidity{timeShifts[i]}h-prior": weatherSet[i][
                        "relative-humidity"
                    ],
                    f"wind-direction{timeShifts[i]}h-prior": weatherSet[i][
                        "wind-direction"
                    ],
                    f"wind-speed{timeShifts[i]}h-prior": weatherSet[i]["wind-speed"],
                }
            )
            for i in range(len(weatherSet))
        ]
        res = pd.concat(res)
        print(f"{row.name}/{floodDf.shape[0]}")
        return res

    # Apply the function to create multiple new columns
    newColumns = floodDf.apply(addWeatherSet, axis=1)
    # Merge the original DataFrame with the new columns
    floodDf = pd.concat([floodDf, newColumns], axis=1)

    return floodDf


def constructDataset(
    predictionTime: int,
    intervalSize: int,
    numReadings: int,
    readingSize: int,
) -> pd.DataFrame:
    """
    Constructs a training dataset based on parameters given.
    """
    pass


# Get data and find nearest station
floodDf, weatherDf = getAllData()
floodDf = calculateClosestStation(floodDf, weatherDf)
floodDf = floodDf.head(1000)

# EXPENSIVE FUNCTION
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
