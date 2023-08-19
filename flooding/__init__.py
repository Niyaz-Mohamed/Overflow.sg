from api import getFlooding


if __name__ == "__main__":
    data = getFlooding()
    data = data[["sensor-id", "sensor-name", "latitude", "longitude"]]
    data.to_csv("stations.csv", index=False)
