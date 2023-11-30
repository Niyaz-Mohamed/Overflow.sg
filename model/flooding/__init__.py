try:
    from .api import getFlooding
    from .db import fetchFromDatabase
except:
    from api import getFlooding
    from db import saveToDatabase
    from datetime import datetime

# Run periodically to save to database
if __name__ == "__main__":
    data = getFlooding()
    print(data)
    now = datetime.now().replace(microsecond=0)
    # Add timestamp to fetched data
    data["timestamp (data fetched)"] = [now for i in range(len(data.index))]
    # Reorder columns
    cols = data.columns.tolist()
    data = data[[cols[-1]] + cols[:-1]]
    saveToDatabase(data)
