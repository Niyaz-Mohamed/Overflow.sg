import os
import pandas as pd

stations = pd.read_csv(os.path.join(__file__, "../stations.csv"))
for index, station in stations.iterrows():
    url = station["URL"]
