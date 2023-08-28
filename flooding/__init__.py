from api import getFlooding
from db import saveToDatabase


if __name__ == "__main__":
    data = getFlooding()
    saveToDatabase(data)
