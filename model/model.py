from data_gen import constructDataset
import pandas as pd, os
from joblib import dump, load

# Import AI related modules
from sklearn import svm
from sklearn.preprocessing import StandardScaler


# Construct training dataset
floodDf = constructDataset(
    predictionTime=1,
    intervalSize=0.5,
    numReadings=3,
    readingSize=10,
    restrictDistance=3,
    restrictRows=2000,
)

# Get sample data
floodDataPath = os.path.join(__file__, "../data/trainingData-1-0.5-3-10.csv")
floodDf = pd.read_csv(floodDataPath)
floodDf["timestamp"] = pd.to_datetime(floodDf["timestamp"])

# Extracting features and target variable
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
labels = floodDf["% full"]

# Scale the features using scikit-learn's StandardScaler
print("scaling...")
scaler = StandardScaler()
scaledFeatures = scaler.fit_transform(features)

# TODO: Test SVM, CART, ANFIS, Random Forest
# Create and train an SVM model (Choose different kernels like 'linear', 'rbf', etc.)
print("modeling")
svm_model = svm.SVR(kernel="rbf")
svm_model.fit(scaledFeatures, labels)
print("complete")

# Save the trained SVM model using joblib, convert to tensorflow with following
# tensorflowjs_converter --input_format=joblib path_to_svm_model.joblib path_to_save_model_directory
savePath = os.path.join(__file__, "../models/svm.joblib")
dump(svm_model, savePath)
# Load the saved SVM model
loaded_svm_model = load(savePath)
