# Extract maximum flood level via OCR, after getting graphs using imager.py
import os, cv2
from statistics import mode, mean
import pandas as pd
import numpy as np

# Make sure tesseractOCR is installed
import pytesseract

# Settings
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
LEFTUPPER = (50, 15)
RIGHTLOWER = (70, 40)
GRAPHTOP = (78, 29)
GRAPHBOTTOM = (78, 436)


# Fetch sensors
sensors = pd.read_csv(os.path.join(__file__, "../sensors.csv"))
maxCapacities = []
errorIndices = []
for index, sensor in sensors.iterrows():
    img = cv2.imread(os.path.join(__file__, f"../images/{sensor['sensor-id']}.png"))

    # Get graph maximum
    digitImg = cv2.resize(
        img[LEFTUPPER[1] : RIGHTLOWER[1], LEFTUPPER[0] : RIGHTLOWER[0]],
        (0, 0),
        fx=3,
        fy=3,
    )
    # if sensor["sensor-id"] == "WWS154":
    #     cv2.imshow("Image", digitImg)
    #     cv2.waitKey(0)

    print(f"OCR {sensor['sensor-id']}: ", end="")
    graphMax = pytesseract.image_to_string(
        digitImg,
        config="--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789.",
    )

    # Handle errors
    try:
        graphMax = float(graphMax)
        print(graphMax)
    except:
        mapper = {
            "WWS154": 6.0,
            "WWS155": 8.0,
            "EWS144": 8.0,
            "WWS319": 6.0,
            "CWS132": 6.0,
            "CWS011": 6.0,
            "WWS415": 6.0,
            "CWS025": 6.0,
            "CWS307": 6.0,
            "CWS043": 8.0,
            "CWS052": 6.0,
            "CWS312": 6.0,
            "EWS013": 6.0,
        }
        graphMax = mapper[sensor["sensor-id"]]
        print(graphMax)

    # Get red line's y coordinate
    redPixels = np.argwhere(cv2.inRange(img, (50, 0, 220), (100, 120, 255)))
    redYCoord = int(mode([p[0] for p in redPixels]))
    # Generate max capacity
    gHeight = GRAPHBOTTOM[1] - GRAPHTOP[1]
    redHeight = gHeight - (redYCoord - GRAPHTOP[1])
    capacity = round(graphMax * (redHeight / gHeight), 2)
    maxCapacities.append(capacity)
    print(f"Capacity: {capacity}m\n")

# Add data to sensors
sensors["max-level"] = maxCapacities
sensors.to_csv(os.path.join(__file__, "../sensors.csv"), index=False)
