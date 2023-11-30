# Document for generation of data
from api import fetchWeatherRange
from myutils import saveWeather, loadWeather

# For testing
if __name__ == "__main__":
    # Generate samples, store in a separate folder
    from datetime import datetime

    date = datetime(year=2023, month=6, day=6)
    df = fetchWeatherRange(date)
    saveWeather(df, "data.csv")
