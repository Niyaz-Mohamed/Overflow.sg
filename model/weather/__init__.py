# Document for generation of data
try:
    from .api import getWeatherRange
except:
    from api import getWeatherRange, getWeather

# For testing
if __name__ == "__main__":
    # Generate samples, store in a separate folder
    from datetime import datetime

    date = datetime(year=2023, month=10, day=3)
    # df = getWeatherRange(date)
    # Remove filtering of empty stations
    thing = getWeather("air-temperature", date)["readings"].drop_duplicates(
        subset="station-id", keep="first"
    )
    print(thing)
