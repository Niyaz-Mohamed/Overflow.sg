# Module for convenience functions
import pandas as pd
from datetime import datetime, timedelta


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


# Expensive function, need to memoize details
memo = {}


def weatherAt(
    dt: datetime, weatherDf: pd.DataFrame, interval: int, stationId: int
) -> pd.Series:
    """
    Calculates weather at a point in time, accounting for \n
    an interval of time around it, given a weather dataset.

    NOTE: Interval should be even
    """

    # Check for memoized results
    key = (dt, interval, stationId)
    if key in memo:
        return memo[key]

    # Create list of minutes to check and filter weatherDf
    readingTimes = [
        dt + (i - interval // 2) * timedelta(minutes=1) for i in range(interval + 1)
    ]
    reading = weatherDf[
        (weatherDf["timestamp"].isin(readingTimes))
        & (weatherDf["station-id"] == stationId)
    ]

    # Group readings in the interval together
    station = reading.iloc[0, :5].squeeze()
    station["timestamp"] = dt
    # Timestamps selected might not be equal to dt, due to 5 minute granularity
    reading = (
        reading.agg(
            {
                "rainfall": "sum",
                "air-temperature": "mean",
                "relative-humidity": "mean",
                "wind-direction": "mean",
                "wind-speed": "mean",
            }
        )
        .squeeze()
        .round(2)
    )
    # Append in station data and memoize
    reading = pd.concat([station, reading])
    memo[key] = reading
    return reading
