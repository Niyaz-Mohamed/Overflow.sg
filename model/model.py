from data_gen import constructDataset
import pandas as pd, os
from joblib import dump, load
from datetime import date

# Import AI related modules
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Construct training dataset
floodDf = constructDataset(
    predictionTime=1,
    intervalSize=0.5,
    numReadings=3,
    readingSize=10,
    restrictDate="2023-11-28",
)


def splitDataset(floodDf):
    """
    Splits the dataset into training and testing, and xTrain and yTrain variables, returning a tuple in this format:\n
    (xTrain, xTest, yTrain, yTest)
    """
    features = floodDf.drop(
        columns=[
            "timestamp",
            "sensor-id",
            "sensor-name",
            "station-id",
            "station-name",
            "station-latitude",
            "station-longitude",
            "station-distance",
            "% full",
        ]
    )
    label = floodDf["% full"]
    xTrain, xTest, yTrain, yTest = train_test_split(features, label, test_size=0.33)
    return xTrain, xTest, yTrain, yTest


def createSVM(floodDf):
    xTrain, xTest, yTrain, yTest = splitDataset(floodDf)
    pass


def createRF(floodDf):
    pass


def createCART(floodDf):
    pass


# # Get sample data
# floodDataPath = os.path.join(__file__, "../data/trainingData-1-0.5-3-10.csv")
# floodDf = pd.read_csv(floodDataPath)
# floodDf["timestamp"] = pd.to_datetime(floodDf["timestamp"])


# # Scale the features using scikit-learn's StandardScaler
# print("scaling...")
# scaler = StandardScaler()
# scaledFeatures = scaler.fit_transform(features)

# # TODO: Test SVM, CART, ANFIS, Random Forest
# # Create and train an SVM model (Choose different kernels like 'linear', 'rbf', etc.)
# print("modeling")
# svm_model = svm.SVR(kernel="rbf")
# svm_model.fit(scaledFeatures, labels)
# print("complete")

# # Save the trained SVM model using joblib, convert to tensorflow with following
# # tensorflowjs_converter --input_format=joblib path_to_svm_model.joblib path_to_save_model_directory
# savePath = os.path.join(__file__, "../models/svm.joblib")
# dump(svm_model, savePath)
# # Load the saved SVM model
# loaded_svm_model = load(savePath)
