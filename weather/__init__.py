# Document for generation of data
from api import fetchWeatherRange
from myutils import saveWeather, loadWeather

# For testing
if __name__ == "__main__":
    from datetime import datetime

    startDate = datetime(year=2023, month=6, day=5)
    endDate = datetime(year=2023, month=6, day=6)

    # Generate samples, store in a separate folder
    df = fetchWeatherRange(startDate, endDate)
    saveWeather(df, "data")
