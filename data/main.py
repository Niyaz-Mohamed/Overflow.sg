# Document for generation of data
from api import fetchWeatherRange
from datetime import datetime

#! del later
from api import fetchWeather
import json

if __name__ == "__main__":
    startDate = datetime(year=2023, month=6, day=5)
    endDate = datetime(year=2023, month=6, day=6)

    # Generate samples, store in a separate folder
    weatherTypes = [
        "air-temperature",
        "relative-humidity",
        "rainfall",
        "wind-direction",
        "wind-speed",
    ]

    df = fetchWeatherRange(startDate, endDate)
    df.to_csv("all-data.csv")
    for type in weatherTypes:
        data = fetchWeather(type, startDate)
        with open(f"{type}.json", "w") as f:
            f.write(json.dumps(data["raw-data"]))
        df = data["readings"]
        df.to_csv(f"{type}.csv")
