import os
from data_gen import constructDataset
import joblib

# Import AI related modules
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib


SAVEPATH = os.path.join(__file__, "../models/")


def saveModel(model, fileName):
    joblib.dump(model, os.path.join(SAVEPATH, fileName))


def testModel(model, xTest, yTest):
    modelPredictions = model.predict(xTest)
    r2 = r2_score(yTest, modelPredictions)
    rmse = mean_squared_error(yTest, modelPredictions) ** (1 / 2)
    mae = mean_absolute_error(yTest, modelPredictions)
    # need to identify where the model parameter is affect? Go beyond using standard metrics. 
    print(f"R-squared (R²) Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")


def splitDataset(floodDf):
    """
    Splits the dataset into training and testing, and xTrain and yTrain variables, returning a tuple in this format:\n
    (xTrain, xTest, yTrain, yTest)
    """
    # Scale the dataset
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
    scaler = StandardScaler()
    # should not scale before splitting the dataset. Split before scaling.
    features = scaler.fit_transform(features)
    # Split into train and test
    xTrain, xTest, yTrain, yTest = train_test_split(features, label, test_size=0.33)
    return xTrain, xTest, yTrain, yTest


def createSVM(xTrain, xTest, yTrain, yTest):
    # Create and test SVM model
    model = SVR(kernel="poly")
    print("SVM\n--------")
    model.fit(xTrain, yTrain)
    testModel(model, xTest, yTest)
    saveModel(model, "SVM.pkl")


def createRF(xTrain, xTest, yTrain, yTest):
    # Create and test random forest
    # how are these hyperparameters tuned? random search?
    # no cross validation?
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    print("RF\n--------")
    model.fit(xTrain, yTrain)
    testModel(model, xTest, yTest)
    saveModel(model, "RF.pkl")


def createCART(xTrain, xTest, yTrain, yTest):
    # Create and test random forest
    model = DecisionTreeRegressor(random_state=42)
    print("CART\n--------")
    model.fit(xTrain, yTrain)
    testModel(model, xTest, yTest)
    saveModel(model, "CART.pkl")
