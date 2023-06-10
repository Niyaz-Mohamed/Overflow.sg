# Document for generation of data
from utils import *

if __name__ == "__main__":
    startDate = datetime(year=2023, month=6, day=5)
    endDate = datetime(year=2023, month=6, day=6)

    df = fetchWeatherRange(startDate, endDate)
    print(df.head(), df.tail())
