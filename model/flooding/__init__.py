try:
    from .db import fetchFromDatabase
except:
    import requests, os
    from db import deleteDuplicates, saveToDatabase, cleanDatabase
    from dotenv import load_dotenv, find_dotenv
    from myutils import parseFlooding


# Run periodically to save to database
if __name__ == "__main__":
    # Fetch raw flooding data
    endpoint = "https://app.pub.gov.sg/waterlevel/pages/GetWLInfo.aspx"
    params = {"type": "WL"}
    response = requests.get(endpoint, params=params)
    data = parseFlooding(response.content.decode("utf-8"))

    # Verify environment variable
    load_dotenv(find_dotenv())
    mongoURI = os.environ.get("MONGODB_URI")
    print(f"\n\nFirst few characters of mongoDB access string {mongoURI[:7]}\n\n")

    # Save modified data to database
    data = data[["timestamp", "sensor-id", "water-level", "status"]]
    print(data, "\n\n", data.info(), end="\n\n")
    saveToDatabase(deleteDuplicates(data))
    cleanDatabase()
