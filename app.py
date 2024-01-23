from model.data_gen import parseFlooding, constructDataset, loadModel

# General purpose modules
import os, pandas as pd, requests
from flask import Flask, request, jsonify, render_template


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
    return render_template("map.html", sensorData=currentData)


@app.route("/update_data", methods=["POST"])
def update_data():
    # Extract data from the POST request
    dateTime = request.form.get("datetime")
    predTime = float(request.form.get("prediction_time"))
    # print(dateTime, predTime, "\n\n\n\n")

    # TODO: Update the data in floodDf based on new values, pull from existing database
    floodDf = getCurrentFloodData()
    floodDf[f"% full ({predTime}h)"] = 80
    # Format the data to send to html file
    floodDf = floodDf.to_dict(orient="records")
    return jsonify(floodDf)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
