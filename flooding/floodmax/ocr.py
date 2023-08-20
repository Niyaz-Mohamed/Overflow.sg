# Extract maximum flood level via OCI
import os
from PIL import Image
import pandas as pd

# Make sure tesseractOCR is installed
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Fetch stations
sensors = pd.read_csv(os.path.join(__file__, "../stations.csv"))
for index, sensor in sensors.iterrows():
    image = Image.open(os.path.join(__file__, f"../images/{sensor['sensor-id']}.png"))
