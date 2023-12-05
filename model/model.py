from data_gen import constructDataset

floodDf = constructDataset(
    predictionTime=1,
    intervalSize=0.5,
    numReadings=3,
    readingSize=10,
    restrictDistance=5,
    restrictRows=500000,
)
