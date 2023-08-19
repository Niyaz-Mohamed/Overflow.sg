from api import getFlooding


if __name__ == "__main__":
    data = getFlooding()
    data = data[["sensor-id", "sensor-name", "latitude", "longitude"]]
    data["URL"] = (
        "https://app.pub.gov.sg/waterlevel/pages/WaterLevelReport.aspx?stationid="
        + data["sensor-id"]
    )
    data.to_csv("stations.csv", index=False)
