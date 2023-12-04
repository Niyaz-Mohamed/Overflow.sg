import os
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from pymongo import MongoClient
from colorama import Fore, Back, Style
from time import time


def saveToDatabase(data: pd.DataFrame):
    """Saves given flood data to a MongoDB database.
    One fetch of flooding data yields about 4.10kB of data"""

    data = list(data.T.to_dict().values())
    # Connect to Mongo, create db and collection
    load_dotenv(find_dotenv())
    mongoURI = os.environ.get("MONGODB_URI")
    client = MongoClient(mongoURI)
    db = client.floodData
    result = db.floodData.insert_many(documents=data)

    # Log results
    # TODO: Add better logging
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
        "\n" + Fore.BLACK + Back.WHITE + "[GET]" + Style.RESET_ALL,
        end=" Getting flooding data from mongo\n",
    )
    fetchTimer = time()
    db = client.floodData
    documents = list(db.floodData.find(query))
    print(
        Fore.BLACK
        + Back.GREEN
        + f"Completed in {round(time()-fetchTimer, 2)}s"
        + Style.RESET_ALL
        + "\n\n"
    )
    return documents
