from colorama import Back, Style
from weather import getWeatherRange, weatherAt
from flooding import fetchFromDatabase
from time import time
from datetime import datetime, timedelta
from geopy.distance import geodesic
import os, joblib, pandas as pd


# Import AI related modules
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def log(
    color: str, foreText: str = "", bodyText: str = "", start: str = "", end: str = "\n"
):
    """
    Convenience function for logging, accepts a color and creates a printed log. Only foreText is colored.

    NOTE: color should be a colorama background color
    """
    print(start + color + foreText + Style.RESET_ALL + " " + bodyText, end=end)


def getAllData(dateRange: list = None):
    """
    Fetches all flooding and relevant weather data and memoizes
    it in a data folder in the same directory as the current file.\n\n
    Returns a floodDf and weatherDf
    """
    # TODO: Make sure only the required flooding data for a given date range is fetched
    # Check if flooding data exists, generate it if otherwise
    floodDataPath = os.path.join(__file__, "../data/floodData.csv")
    # Load data if it does not exist
    if not os.path.isfile(floodDataPath):
        if dateRange:
            floodDf = fetchFromDatabase(
                query={
                    "status": {"$in": [0, 1, 2]},
                    "timestamp": {
                        "$gte": dateRange[0],
                        "$lte": dateRange[1] + " 23:59:59",
                    },
                }
            )
        else:
            floodDf = fetchFromDatabase(
                query={
                    "status": {"$in": [0, 1, 2]},
                }
            )
        floodDf.to_csv(floodDataPath, index=False)
    # Check if existing dataset falls within date range
    elif dateRange:
        # Get the start and end dates
        floodDf = pd.read_csv(floodDataPath)
        floodDf["timestamp"] = pd.to_datetime(floodDf["timestamp"])
        dataStartDate = floodDf["timestamp"].min()
        dataEndDate = floodDf["timestamp"].max()

        # Convert provided date range strings to datetime
        providedStartDate = pd.to_datetime(dateRange[0] + "T00:00:00")
        providedEndDate = pd.to_datetime(dateRange[1] + "T23:59:59")

        # Check if data's start and end dates fall within the provided date range
        startDateWithinRange = providedStartDate <= dataStartDate <= providedEndDate
        endDateWithinRange = providedStartDate <= dataEndDate <= providedEndDate

        if not (startDateWithinRange and endDateWithinRange):
            addedData = fetchFromDatabase(
                query={
                    "status": {"$in": [0, 1, 2]},
                    "timestamp": {"$gte": dateRange[0], "$lte": dateRange[1]},
                }
            )
            floodDf = pd.concat([floodDf, addedData]).drop_duplicates(
                subset=["timestamp", "sensor-id"], keep="first"
            )
            floodDf.to_csv(floodDataPath, index=False)

    # Load memoized flooding data
    floodDf = pd.read_csv(floodDataPath)

    # Check if weather data is suitable for flood data
    weatherDataPath = os.path.join(__file__, "../data/weatherData.csv")
    floodMin = datetime.fromisoformat(floodDf["timestamp"].min()).date()
    floodMax = datetime.fromisoformat(floodDf["timestamp"].max()).date()

    # Generate data if it doesn't exist
    if not os.path.isfile(weatherDataPath):
        weatherDf = getWeatherRange(floodMin - timedelta(days=1), floodMax)
        weatherDf.to_csv(weatherDataPath, index=False)
        weatherDf = pd.read_csv(weatherDataPath)
    # Define conditions on which to add more dates
    else:
        weatherDf = pd.read_csv(weatherDataPath)
        weatherMin = datetime.fromisoformat(weatherDf["timestamp"].min()[:-6]).date()
        weatherMax = datetime.fromisoformat(weatherDf["timestamp"].max()[:-6]).date()
        prependDates = weatherMin + timedelta(days=1) > floodMin
        appendDates = weatherMax < floodMax

        # Prepend missing early dates
        if prependDates:
            prependDf = getWeatherRange(floodMin - timedelta(days=1), weatherMin)
            weatherDf = pd.concat([prependDf, weatherDf])

        # Append missing end dates
        if appendDates:
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
    # Set columns required
    floodDf = floodDf[
        [
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
            "status",
        ]
    ]

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
    `readingSize`: Size of period around reading time (in hours), taken to be reading for that time\n
    """

    # Set up timer
    log(
        Back.CYAN,
        "[TASK]",
        "Injecting weather data into flooding dataset\n",
    )
    startTime = time()

    # Function to generate weather columns for a particular time shift (iterates through df)
    def addWeatherSet(row, timeShift):
        # Log progress every 5000 rows
        if (row.name + 1) % 5000 == 0:
            log(
                Back.WHITE,
                "[UPDATE]",
                f"Completed {row.name+1}/{fillerDf.shape[0]} rows ({round(time()-startTime)}s)",
            )

        # Set timestamp at which weather is measured ()
        try:
            timestamp = row["timestamp"].to_pydatetime()
        except:
            timestamp = datetime.fromisoformat(row["timestamp"].replace(" ", "T"))

        # Calculate weather for given timeshift
        weatherSeries = weatherAt(
            timestamp - timedelta(hours=timeShift),
            weatherDf,
            interval=round(readingSize * 60),
            stationId=row["station-id"],
        )

        # Rename the resultant weather
        weatherSeries = weatherSeries[
            [
                "rainfall",
                "air-temperature",
                "relative-humidity",
                "wind-direction",
                "wind-speed",
            ]
        ]
        weatherSeries.index = weatherSeries.index.map(
            lambda x: x + f"-{timeShift}h-prior"
        )
        return weatherSeries

    # Iterate through timeshifts
    timeShifts = [predictionTime + n * intervalSize for n in range(numReadings)]
    for timeShift in timeShifts:
        # Filter rows to fill based on existence of the columns
        testColumn = f"rainfall-{timeShift}h-prior"
        if testColumn in floodDf.columns:
            # Filter only nan rows to fill
            fillerDf = floodDf[pd.isnull(floodDf[testColumn])].reset_index(drop=True)
        else:
            # Else use entire dataset
            fillerDf = floodDf.copy(deep=True)

        # Check for and logs whether to continue
        if fillerDf.shape[0] == 0:
            log(
                Back.GREEN,
                "[FETCH]",
                f"No updates for timeshift of -{timeShift}h",
            )
            continue
        else:
            log(
                Back.GREEN,
                "[FETCH]",
                f"Weather for timeshift of -{timeShift}h ({fillerDf.shape[0]} rows)",
            )

        # Filter essential info necessary for filling weather data
        fillerDf = fillerDf[["timestamp", "sensor-id", "station-id"]]

        # Insert weather data based on time shift
        newColumns = fillerDf.apply(addWeatherSet, axis=1, timeShift=timeShift)
        fillerDf = pd.concat([fillerDf, newColumns], axis=1)

        # Update flood dataframe if columns required are already present
        if testColumn in floodDf.columns:
            floodDf.set_index(["timestamp", "sensor-id", "station-id"], inplace=True)
            fillerDf.set_index(["timestamp", "sensor-id", "station-id"], inplace=True)
            floodDf.update(fillerDf)
            floodDf.reset_index(inplace=True)
        # Update floodDf if columns required not present
        else:
            fillerDf.drop(
                columns=["timestamp", "sensor-id", "station-id"], inplace=True
            )
            floodDf = pd.concat([floodDf, fillerDf], axis=1)

        log(
            Back.GREEN,
            "[COMPLETE]",
            f"Finished weather timeshift of -{timeShift}h ({round(time()-startTime)}s) \n",
        )

    # Log time and return
    log(Back.GREEN, f"Completed in {round((time()-startTime)/60,2)}min", "\n")
    return floodDf


def constructDataset(
    predictionTime: int,
    intervalSize: int = 0.5,
    readingSize: int = 0.5,
    numReadings: int = 3,
    restrictDistance: int = None,
    restrictDate: list = None,
) -> pd.DataFrame:
    """
    Constructs a training dataset based on parameters given.\n\n
    Returns a training dataset which it will also memoize in a data folder.

    Parameters
    -----------
    `predictionTime`: Prediction time (in hours) that the AI produced will have\n
    `intervalSize`: Length of interval (in hours) between each moment in time where weather is measured\n
    `readingSize`: Size of period around reading time (in hours), taken to be reading for that time\n
    `numreadings`: Number of moments in time where weather is measured (prior to prediction time)\n
    `restrictDistance`: Only accept rows where station-to-sensor distance is lower than this [optional]\n
    `restrictDate`: 1 or 2 date values to filter dates from start to end time, formatted as '2023-01-30'\n
    """

    # Fetch base datasets
    log(Back.CYAN, "[TASK]", "Fetching and saving base weather and flooding datasets")
    floodDf, weatherDf = getAllData(restrictDate)

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
            startDate = pd.to_datetime(restrictDate[0] + "T00:00:00")
            endDate = pd.to_datetime(restrictDate[1] + "T23:59:59")

        # Apply filters
        floodDf = floodDf[
            (floodDf["timestamp"] >= startDate) & (floodDf["timestamp"] <= endDate)
        ].reset_index(drop=True)

    # Match water level sensors to closest weather station
    log(Back.CYAN, "[TASK]", "Matching weather stations to flood sensors")
    floodDf = calculateClosestStation(floodDf, weatherDf)

    # Check for existing full dataset and load it in
    savePath = os.path.join(__file__, f"../data/trainingData.csv")
    if os.path.isfile(savePath):
        log(
            Back.GREEN,
            "[INFO]",
            "Existing dataset found, updating dataset with required data",
        )
        log(
            Back.RED,
            "[WARN]",
            "If reading size is different for this run than when existing dataset was first created, please remake the dataset",
        )
        existingDf = pd.read_csv(savePath)
        existingDf["timestamp"] = pd.to_datetime(existingDf["timestamp"])

        # Combine existing dataframe to current flooding dataframe (nulls to unknown values)
        floodDf = pd.concat([existingDf, floodDf], ignore_index=True)
        floodDf["timestamp"] = pd.to_datetime(floodDf["timestamp"])
        floodDf = (
            floodDf.drop_duplicates(subset=["timestamp", "sensor-id"])
            .reset_index(drop=True)
            .sort_values(by="timestamp")
        )

    # Inject weather data into flooding data based on factors (EXPENSIVE OPERATION)
    floodDf = injectWeatherData(
        floodDf,
        weatherDf,
        predictionTime,
        intervalSize,
        numReadings,
        readingSize,
    )

    # Get list of columns to keep
    timeShifts = [predictionTime + n * intervalSize for n in range(numReadings)]
    weatherColumns = [
        [
            f"rainfall-{timeShift}h-prior",
            f"relative-humidity-{timeShift}h-prior",
            f"air-temperature-{timeShift}h-prior",
            f"wind-speed-{timeShift}h-prior",
            f"wind-direction-{timeShift}h-prior",
        ]
        for timeShift in timeShifts
    ]
    weatherColumns = [element for sublist in weatherColumns for element in sublist]

    # Fill NaNs (DO NOT CHANGE REPLACEMENT NUMBER)
    replacementNumber = -99999
    floodDf[weatherColumns] = floodDf[weatherColumns].fillna(replacementNumber)

    # Save the dataset
    savePath = os.path.join(__file__, f"../data/trainingData.csv")
    floodDf = floodDf.sort_values(by="timestamp")
    floodDf.to_csv(savePath, index=False)

    # Remove perviously filled NaNs
    nanRowsMask = floodDf[weatherColumns].eq(replacementNumber).any(axis=1)
    floodDf = floodDf[~nanRowsMask]

    # Filter columns needed
    newColumns = list(floodDf.columns[:12]) + weatherColumns
    floodDf = floodDf[newColumns]

    # Restrict date range for full dataset
    if restrictDate:
        floodDf["timestamp"] = pd.to_datetime(floodDf["timestamp"])
        # Apply filters
        floodDf = floodDf[
            (floodDf["timestamp"] >= startDate) & (floodDf["timestamp"] <= endDate)
        ].reset_index(drop=True)

    # Restrict distance
    if restrictDistance:
        log(
            Back.CYAN,
            "[TASK]",
            f"Restricting station-distance in flood dataset to below {restrictDistance}km",
        )
        floodDf = floodDf[floodDf["station-distance"] < restrictDistance]
        floodDf.reset_index(drop=True, inplace=True)

    # Log and save current dataset
    log(
        Back.GREEN,
        "[INFO]",
        f"Produced dataset with {floodDf.shape[0]} rows\n",
    )
    floodDf.to_csv(os.path.join(__file__, f"../data/currentTrainingData.csv"))
    return floodDf


# AI Related Functions Below
def saveModel(model, fileName):
    """
    Ensure filename includes file extension.
    """
    saveDir = os.path.join(__file__, "../models/")
    if not os.path.isdir(saveDir):
        os.makedirs(saveDir)
    joblib.dump(model, os.path.join(saveDir, fileName))


def splitDataset(floodDf):
    """
    Splits the dataset into training and testing, and xTrain and yTrain variables, returning a tuple in this format:\n
    (xTrain, xTest, yTrain, yTest)
    """
    # Scale the dataset
    features = floodDf.drop(
        columns=[
            "timestamp",
            "sensor-id",
            "sensor-name",
            # "sensor-latitude",
            # "sensor-longitude",
            "station-id",
            "station-name",
            "station-latitude",
            "station-longitude",
            "station-distance",
            "% full",
        ]
    )
    label = floodDf["% full"]

    # Split into train and test
    xTrain, xTest, yTrain, yTest = train_test_split(features, label, train_size=0.80)

    # Scale and return data after split
    scaler = MinMaxScaler()
    xTrain = scaler.fit_transform(xTrain)
    xTest = scaler.fit_transform(xTest)
    return xTrain, xTest, yTrain, yTest
