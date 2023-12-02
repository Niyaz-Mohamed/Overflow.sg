# Document for generation of data
try:
    from .api import getWeatherRange
    from .myutils import weatherAt
except:
    from api import getWeatherRange, getWeather

# For testing
if __name__ == "__main__":
    # Generate samples, store in a separate folder
    from datetime import datetime

    date = datetime(year=2023, month=10, day=3)
    df = getWeatherRange(date)
    print(df)
