from api import getFlooding
from db import saveToDatabase

# Run periodically to save to database
if __name__ == "__main__":
    data = getFlooding()
    saveToDatabase(data)
