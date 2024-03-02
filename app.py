#########################################################
#########################################################
#########################################################
#########################################################
#########################################################
#########################################################
#########################################################
#! THIS WEBSITE IS A DEMO AND DOES NOT DYNAMICALLY UPDATE DO NOT USE THIS AS AN ACTUAL FLOODING DETERMINANT

from model.data_gen import parseFlooding, constructDataset, loadModel

# General purpose modules
import os, pandas as pd, requests
from datetime import timedelta
from flask import Flask, request, jsonify, render_template

# Create Flask app
app = Flask(__name__)
# Load dataframe for later use
demoDf = pd.read_csv(os.path.join(__file__, "../static/DemoData.csv"))
demoDf["timestamp"] = pd.to_datetime(demoDf["timestamp"])
demoDf = demoDf.sort_values(by="timestamp")
# Find date range of the demo data
dateRange = [demoDf["timestamp"].min(), demoDf["timestamp"].max()]
dateRange = [date.strftime("%Y-%m-%d %H:%M:%S") for date in dateRange]


@app.route("/")
def map():
    # Convert the DataFrame to a list of dictionaries
    firstReadings = demoDf.drop_duplicates(subset=["sensor-id"])
    firstReadings = firstReadings.to_dict(orient="records")
    return render_template("map.html", sensorData=firstReadings, dateRange=dateRange)


@app.route("/update_data", methods=["POST"])
def update_data():
    # Extract data from the POST request
    dateTime = pd.to_datetime(request.form.get("datetime"))
    predTime = float(request.form.get("prediction_time"))
    print("Date Time: ", dateTime)
    print("Prediction Lead Time: ", predTime)

    # Select rows before datetime, that are closest to that datetime
    floodDf = demoDf.copy(deep=True)
    floodDf = floodDf[floodDf["timestamp"] <= dateTime]
    floodDf = floodDf.sort_values(by="timestamp").drop_duplicates(
        subset="sensor-id", keep="last"
    )

    # Format the data to send to html file
    floodDf = floodDf.to_dict(orient="records")
    return jsonify(floodDf)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
