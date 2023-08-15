# Document for generation of data
from api import fetchWeatherRange
from datetime import datetime

if __name__ == "__main__":
    startDate = datetime(year=2023, month=6, day=6)
    endDate = datetime(year=2023, month=6, day=6)

    # Generate samples, store in a separate folder
    df = fetchWeatherRange(startDate, endDate)
    df.to_csv("data.csv", index=False)
