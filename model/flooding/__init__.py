try:
    from .db import fetchFromDatabase
except:
    import requests
    from db import deleteDuplicates, saveToDatabase, cleanDatabase, fetchFromDatabase
    from myutils import parseFlooding


# Run periodically to save to database
if __name__ == "__main__":
    # TODO: Test if AI wasn't broken by this
    # TODO: Also remove max-level during flood data fetching, and include status within final training dataset, but remove from actual training

    # Fetch raw flooding data
    endpoint = "https://app.pub.gov.sg/waterlevel/pages/GetWLInfo.aspx"
    params = {"type": "WL"}
    response = requests.get(endpoint, params=params)
    data = parseFlooding(response.content.decode("utf-8"))

    # Save modified data to database
    data = data[["timestamp", "sensor-id", "water-level", "status"]]
    print(data, "\n\n", data.info(), end="\n\n")
    saveToDatabase(deleteDuplicates(data))
    cleanDatabase()
