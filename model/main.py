from data_gen import constructDataset
from train import splitDataset, createSVM, createRF, createCART

# Construct training dataset
floodDf = constructDataset(
    predictionTime=1,
    intervalSize=0.5,
    numReadings=3,
    readingSize=10,
    restrictDate="2023-11-28",
)

# Construct models
splitData = splitDataset(floodDf)
createSVM(*splitData)
createRF(*splitData)
createCART(*splitData)
