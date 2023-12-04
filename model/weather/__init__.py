# Document for generation of data
try:
    from .api import getWeatherRange
    from .myutils import weatherAt
except:
    from api import getWeatherRange
    from datetime import datetime

# For testing weather fetching
if __name__ == "__main__":
    # Get weather range
    date = datetime(year=2023, month=10, day=3)
    df = getWeatherRange(date)
    df = df.sort_values(by=["timestamp", "station-id"])

    # Clean and save
    df = df.dropna()
    df.to_csv("data.csv", index=False)
