import os, pandas as pd
from datetime import datetime, timedelta
from pymongo import MongoClient
from dotenv import load_dotenv, find_dotenv
from time import time
from colorama import Fore, Back, Style


def saveToDatabase(data: pd.DataFrame):
    """
    Saves given flood data to a MongoDB database.
    One save of flooding data yields about 4.10kB of data.
    """

    data = list(data.T.to_dict().values())
    # Connect to Mongo, create db and collection
    load_dotenv(find_dotenv())
    mongoURI = os.environ.get("MONGODB_URI")
    client = MongoClient(mongoURI)
    db = client.floodData
    result = db.floodData.insert_many(documents=data)
    client.close()

    # Log results
    if result:
        print(f"Database write successful, wrote {len(data)} documents to collection")


def fetchFromDatabase(query=None):
    """Fetches all flooding data from the MongoDB database.
    Optionally provide a query to filter results"""

    # Connect to Mongo
    load_dotenv(find_dotenv())
    mongoURI = os.environ.get("MONGODB_URI")
    client = MongoClient(mongoURI)

    # Get collection data
    print(
        Fore.BLACK + Back.WHITE + "[GET]" + Style.RESET_ALL,
        end=" Getting flooding data from mongo\n",
    )
    fetchTimer = time()
    db = client.floodData
    floodDf = pd.DataFrame(list(db.floodData.find(query))).drop(columns=["_id"])

    # Append required data to the dataset
    sensorPath = os.path.abspath(os.path.join(__file__, "../floodmax/sensors.csv"))
    sensors = pd.read_csv(sensorPath)
    floodDf = floodDf.merge(sensors, on="sensor-id").sort_values(
        by="timestamp", ascending=True
    )
    floodDf[r"% full"] = round((floodDf["water-level"] / floodDf["max-level"]) * 100, 2)
    floodDf = floodDf[
        [
            "timestamp",
            "sensor-id",
            "sensor-name",
            "latitude",
            "longitude",
            "max-level",
            "% full",
            "status",
        ]
    ]

    # Close the connection and return the data
    client.close()
    print(
        Fore.BLACK
        + Back.GREEN
        + f"Completed in {round(time()-fetchTimer, 2)}s"
        + Style.RESET_ALL
    )
    return floodDf


def deleteDuplicates(floodDf):
    """
    Deletes all entries in a flood dataframe that are already present in the mongo database.
    """
    # Connect to MongoDB
    load_dotenv(find_dotenv())
    mongoURI = os.environ.get("MONGODB_URI")
    client = MongoClient(mongoURI)
    db = client.floodData
    collection = db.floodData

    # Get existing composite keys from MongoDB
    existingKeysCursor = collection.find({}, {"timestamp": 1, "sensor-id": 1, "_id": 0})
    existingKeys = set(
        (doc["timestamp"], doc["sensor-id"]) for doc in existingKeysCursor
    )
    # Filter and return DataFrame based on existing keys
    origLength = floodDf.shape[0]
    floodDf = floodDf[
        ~floodDf.apply(
            lambda row: (row["timestamp"], row["sensor-id"]) in existingKeys, axis=1
        )
    ]
    print(
        f"{origLength-floodDf.shape[0]} duplicate entries excluded from database write"
    )
    client.close()
    return floodDf


def cleanDatabase():
    """
    Cleans the database, deleting entries older than 1yo+
    """
    # Connect to Mongo, create db and collection
    load_dotenv(find_dotenv())
    mongoURI = os.environ.get("MONGODB_URI")
    client = MongoClient(mongoURI)
    db = client.floodData
    collection = db.floodData

    # Calculate the timestamp threshold (1 year ago from the current time)
    twoYearsAgo = datetime.now() - timedelta(days=731)
    timestampThreshold = twoYearsAgo.isoformat()  # Convert to ISO format string

    # Define the deletion query and delete
    query = {"timestamp": {"$lt": timestampThreshold}}
    result = collection.delete_many(query)
    print(f"Deleted {result.deleted_count} documents from more than 2 years ago")

    # Close the MongoDB connection
    client.close()
