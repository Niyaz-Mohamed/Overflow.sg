from model.data_gen import parseFlooding, constructDataset
from model import loadModel

# General purpose modules
import os, pandas as pd, requests
from flask import Flask, render_template


# Create Flask app
app = Flask(__name__)


def getCurrentFloodData():
    """
    Fetches raw current flooding data
    """
    # Pull current raw flooding data
    endpoint = "https://app.pub.gov.sg/waterlevel/pages/GetWLInfo.aspx"
    params = {"type": "WL"}
    response = requests.get(endpoint, params=params)
    floodDf = parseFlooding(response.content.decode("utf-8"))[
        ["timestamp", "sensor-id", "water-level", "status"]
    ]
    # Inject more data into dataframe
    sensorPath = os.path.abspath(
        os.path.join(__file__, "../model/flooding/floodmax/sensors.csv")
    )
    sensors = pd.read_csv(sensorPath)
    floodDf = floodDf.merge(sensors, on="sensor-id").sort_values(
        by="timestamp", ascending=True
    )
    floodDf[r"% full"] = round((floodDf["water-level"] / floodDf["max-level"]) * 100, 2)
    return floodDf


@app.route("/")
def map():
    # Convert the DataFrame to a list of dictionaries
    currentData = getCurrentFloodData().to_dict(orient="records")
    return render_template("map.html", sensors=currentData)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
