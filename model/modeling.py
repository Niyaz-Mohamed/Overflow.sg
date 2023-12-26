import os, joblib, pandas as pd
from data_gen import constructDataset, log
from colorama import Back


# Import AI related modules
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
from xgboost import XGBRegressor


SAVEPATH = os.path.join(__file__, "../models/")


def saveModel(model, fileName):
    joblib.dump(model, os.path.join(SAVEPATH, fileName))


def testModel(model, xTest, yTest, epsilon=10e-3):
    modelPredictions = model.predict(xTest)
    r2 = r2_score(yTest, modelPredictions)
    rmse = mean_squared_error(yTest, modelPredictions) ** (1 / 2)
    mae = mean_absolute_error(yTest, modelPredictions)
    # Add an epsilon to reduce huge values
    mape = mean_absolute_percentage_error(yTest + epsilon, modelPredictions)

    # TODO: Need to identify where the model parameter is affected? Go beyond using standard metrics.
    log(Back.WHITE, "R²:", f"{r2:.4f}")
    log(Back.WHITE, "RMSE:", f"{rmse:.4f}")
    log(Back.WHITE, "MAE:", f"{mae:.4f}")
    log(Back.WHITE, "MAPE:", f"{mape:.4f}\n")

    return {
        "R²": round(r2, 2),
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "MAPE": round(mape, 2),
    }


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
            # "sensor-latitude",
            # "sensor-longitude",
            "station-id",
            "station-name",
            "station-latitude",
            "station-longitude",
            "station-distance",
            "% full",
        ]
    )
    label = floodDf["% full"]

    # Split into train and test
    xTrain, xTest, yTrain, yTest = train_test_split(features, label, train_size=0.80)

    # Scale and return data after split
    scaler = StandardScaler()
    xTrain = scaler.fit_transform(xTrain)
    xTest = scaler.fit_transform(xTest)
    return xTrain, xTest, yTrain, yTest


def createSVM(xTrain, xTest, yTrain, yTest):
    # Create SVM model
    model = SVR(kernel="poly")
    log(Back.CYAN, "__________SVM__________", end="\n")
    model.fit(xTrain, yTrain)

    # Perform testing
    eval = testModel(model, xTest, yTest)
    saveModel(model, "SVM.pkl")
    return model, eval


def createCART(xTrain, xTest, yTrain, yTest):
    # Create CART
    model = DecisionTreeRegressor(random_state=42)
    log(Back.CYAN, "__________CART__________", end="\n")
    model.fit(xTrain, yTrain)

    # Perform testing
    eval = testModel(model, xTest, yTest)
    saveModel(model, "CART.pkl")
    return model, eval


def createRF(xTrain, xTest, yTrain, yTest):
    # Create random forest
    # how are these hyperparameters tuned? random search?
    # no cross validation?
    model = RandomForestRegressor(random_state=42)
    log(Back.CYAN, "__________RF__________", end="\n")
    model.fit(xTrain, yTrain)

    # Perform testing
    eval = testModel(model, xTest, yTest)
    saveModel(model, "RF.pkl")
    return model, eval


def createXGB(xTrain, xTest, yTrain, yTest):
    # Create XGBoost
    model = XGBRegressor()
    log(Back.CYAN, "__________XGB__________", end="\n")
    model.fit(xTrain, yTrain)

    # Perform testing
    eval = testModel(model, xTest, yTest)
    saveModel(model, "XGB.pkl")
    return model, eval


def createMLP(xTrain, xTest, yTrain, yTest):
    # Create MLP
    model = MLPRegressor(random_state=42)
    log(Back.CYAN, "__________MLP__________", end="\n")
    model.fit(xTrain, yTrain)

    # Perform testing
    eval = testModel(model, xTest, yTest)
    saveModel(model, "MLP.pkl")
    return model, eval


def trialModels(predictionTimes: list, dateRange: list, restrictDistance: int = None):
    """
    Evaluates all avaiable models for different prediction times (in hours) given, using flooding data from given date range.

    NOTE: Function must be updated when new models are added
    """

    trialDf = None
    # List of models to test (Approppriate model creation function needed)
    models = ["RF", "MLP"]

    # Iterate through parameters for dataset
    for predTime in predictionTimes:
        log(Back.GREEN, f"MODELS FOR PREDICTION TIME {predTime}h", "\n")
        floodDf = constructDataset(
            predictionTime=predTime,
            restrictDate=dateRange,
            restrictDistance=restrictDistance,
        )

        # Construct result dict to update with model test results
        result = [
            {
                "prediction-time": predTime,
                "model": models[i],
                "R²": 0,
                "MAE": 0,
                "RMSE": 0,
                "MAPE": 0,
            }
            for i in range(len(models))
        ]
        # Test all models needed
        splitData = splitDataset(floodDf)
        for i in range(len(result)):
            model = result[i]["model"]
            # Get and save model results
            modelResults = eval(f"create{model}(*splitData)")[1]
            result[i].update(modelResults)

        # Update the dataframe containing results for all prediction times
        try:
            trialDf = pd.concat([trialDf, pd.DataFrame(result)]).reset_index(drop=True)
        except:
            trialDf = pd.DataFrame(result)

    trialDf.to_csv(os.path.join(SAVEPATH, "eval.csv"), index=False)
