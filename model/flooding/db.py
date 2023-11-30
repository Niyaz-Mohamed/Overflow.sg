import os
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from pymongo import MongoClient


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

    # Connect to Mongo, get collection data
    load_dotenv(find_dotenv())
    mongoURI = os.environ.get("MONGODB_URI")
    client = MongoClient(mongoURI)
    db = client.floodData
    documents = list(db.floodData.find(query))
    return documents
