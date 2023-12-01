# Module for convenience functions
import pandas as pd
from datetime import datetime, timedelta
from statistics import mean


def normalizeWeather(data: dict) -> pd.DataFrame:
    """
    Normalizes and cleans weather readings (in dictionary format), converting it to a flattened dataframe.
    """
    # Flatten json
    df = pd.json_normalize(data, record_path="readings", meta=["timestamp"])
    # Clean up by reordering and sorting columns
    cols = df.columns.to_list()
    cols = cols[-1:] + cols[:-1]
    df = df[cols].sort_values(by=["timestamp", "station_id"])
    df = df.rename(columns={"station_id": "station-id"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%dT%H:%M:%S%z")
    return df.reset_index(drop=True)


def normalizeStations(data: dict) -> pd.DataFrame:
    """
    Normalizes and cleans station data (in dictionary format), converting it to a flattened dataframe.
    """
    # Flatten json and clean df
    df = pd.json_normalize(data)
    df = df.drop(columns="device_id").rename(
        columns={
            "id": "station-id",
            "name": "station-name",
            "location.latitude": "latitude",
            "location.longitude": "longitude",
        }
    )
    return df


def mergeWeather(
    data: list[pd.DataFrame], how: str = "left", on: list = ["timestamp", "station-id"]
) -> pd.DataFrame:
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


def concatStations(data: list[pd.DataFrame]) -> pd.DataFrame:
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


def saveWeather(weatherDf: pd.DataFrame, name: str):
    """
    Convenience function to save a weather dataframe to the file name.
    """
    weatherDf.to_csv(name, index=False)


def loadWeather(name):
    """
    Loads a weather dataframe from [name].csv
    """
    df = pd.read_csv(f"{name}.csv").infer_objects()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def weatherAt(
    dt: datetime, weatherDf: pd.DataFrame, interval: int, stationId: int
) -> pd.DataFrame:
    """
    Calculates weather at a point in time, accounting for \n
    an interval of time around it, given a weather dataset.

    NOTE: Interval should be even
    """

    # Create list of minutes to check and filter weatherDf
    readingTimes = [
        dt + (i - interval // 2) * timedelta(minutes=1) for i in range(interval + 1)
    ]
    reading = weatherDf[
        (weatherDf["timestamp"].isin(readingTimes))
        & (weatherDf["station-id"] == stationId)
    ]

    # Group readings in the interval together
    reading = (
        reading[reading["timestamp"] == dt]
        .agg(
            {
                "rainfall": "sum",
                "air-temperature": "mean",
                "relative-humidity": "mean",
                "wind-direction": "mean",
                "wind-speed": "mean",
            }
        )
        .squeeze()
    )
    # Append in station data
    station = reading.iloc[:1, :5].squeeze()
    reading = pd.DataFrame(pd.concat([station, reading])).T
    return reading
