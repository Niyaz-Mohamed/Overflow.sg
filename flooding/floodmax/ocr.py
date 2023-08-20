# Extract maximum flood level via OCI
import os
from statistics import mode
import cv2
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


# Fetch stations
sensors = pd.read_csv(os.path.join(__file__, "../stations.csv"))
maxlevels = []
for index, sensor in sensors.iterrows():
    img = cv2.imread(os.path.join(__file__, f"../images/{sensor['sensor-id']}.png"))

    # Get graph maximum
    digitImg = cv2.resize(
        img[LEFTUPPER[1] : RIGHTLOWER[1], LEFTUPPER[0] : RIGHTLOWER[0]],
        (0, 0),
        fx=3,
        fy=3,
    )
    print(f"OCR {sensor['sensor-id']}: ", end="")

    graphMax = float(
        pytesseract.image_to_string(
            digitImg,
            config="--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789.",
        )
    )
    print(graphMax)
    # cv2.imshow("Image", digitImg)
    # cv2.waitKey(0)

    # Get sensor capacity
    red = np.where(img == [231, 76, 60])
    ycoords = list(red[0])
    # Remove extremes
