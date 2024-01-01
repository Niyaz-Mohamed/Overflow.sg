from data_gen import constructDataset, saveModel, log
from pprint import PrettyPrinter
import numpy as np, pandas as pd, os
from colorama import Back

# Import AI related modules
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (
    make_scorer,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

#! Model Settings
MODEL = "CART"  # Model to test
DATERANGE = ["2023-11-26", "2023-11-30"]  # Date range to train over
PREDTIME = 2  # Prediction time for the model
STATIONDISTANCE = 3.5  # Maximum Sensor-Station Distance Accepted
EPSILON = 1e-10  # Value of epsilon to use for MAPE calculations

#! Print Settings
PRINTALL = False  # Whether to print all raw results or to only give important results
SAVERESULTS = False  # Whether to save data from splits. Only used if PRINTALL is False

# Define grid of model hyperparameters
paramGrid = {
    # SVR parameters
    "SVM": {
        "C": [0.9],
    },
    # CART parameters
    "CART": {
        # "max_depth": [None],
        "min_samples_split": [200],
        "min_samples_leaf": [60],
        # "max_features": [None],
    },
    # RF parameters
    "RF": {
        "n_estimators": [200],
        "min_samples_split": [5],
        "min_samples_leaf": [20],
    },
    # XGB parameters
    "XGB": {
        "n_estimators": [100],
        "max_depth": [8],
        "learning_rate": [0.1],
        "subsample": [0.7],
        "gamma": [2],
        "scale_pos_weight": [0],  # Default 1
        "alpha": [10],  # Default 0
        "lambda": [0],  # Default 1
    },
    # MLP parameters
    "MLP": {
        "activation": ["tanh"],
        "alpha": [0.001],
        "learning_rate_init": [0.01],
    },
}

# Define grid of models
modelGrid = {
    "SVM": SVR(),
    "CART": DecisionTreeRegressor(),
    "RF": RandomForestRegressor(),
    "XGB": XGBRegressor(),
    "MLP": MLPRegressor(),
}


# Define MAPE and RMSE scorers
def mean_absolute_percentage_error_eps(y_true, y_pred, epsilon=EPSILON):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon)))


def root_mean_squared_error(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)


# Create custom scorers using make_scorer
scoring = {
    "r2": make_scorer(r2_score),
    "neg_MAE": make_scorer(mean_absolute_error, greater_is_better=False),
    "neg_RMSE": make_scorer(root_mean_squared_error, greater_is_better=False),
    "neg_MAPE": make_scorer(
        mean_absolute_percentage_error_eps, greater_is_better=False
    ),
}

# Prepare exhaustive
gridSearch = GridSearchCV(
    estimator=modelGrid[MODEL],
    param_grid=paramGrid[MODEL],
    cv=5,
    scoring=scoring,
    refit="r2",
    verbose=3,
    n_jobs=-1,  # Use all available CPU cores
)

# Get data and preprocess it
data = constructDataset(
    predictionTime=PREDTIME, restrictDate=DATERANGE, restrictDistance=STATIONDISTANCE
)
data = data.sample(frac=1).reset_index(drop=True)  # Shuffle data
x = data.drop(
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
y = data["% full"]
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

# Fit data to grid search
gridSearch.fit(x, y)
model = gridSearch.best_estimator_
saveModel(model, f"{MODEL}-{PREDTIME}h-{STATIONDISTANCE}km.pkl")

# Access best params
log(Back.GREEN, "BEST PARAMETERS", "\n", start="\n")
PrettyPrinter().pprint(gridSearch.best_params_)
results = gridSearch.cv_results_
bestIndex = gridSearch.best_index_

if not PRINTALL:
    bestResults = pd.Series(
        data=[
            results["mean_test_r2"][bestIndex],
            -results["mean_test_neg_MAE"][bestIndex],
            -results["mean_test_neg_RMSE"][bestIndex],
            -results["mean_test_neg_MAPE"][bestIndex],
            results["mean_fit_time"][bestIndex],
            results["mean_score_time"][bestIndex],
        ],
        index=["R2", "MAE", "RMSE", "MAPE", "Fit Time", "Score Time"],
    )

    # Access best evaluations
    log(Back.GREEN, "MEAN EVALS FOR BEST PARAMETERS", "\n", start="\n")
    print(bestResults)

    # Keep data in a dataframe in case you need to save it
    log(Back.GREEN, "SPLIT EVALUATIONS", "\n", start="\n")
    resDf = pd.DataFrame(columns=["SplitNo", "R2", "MAE", "RMSE", "MAPE"])

    # Get and savesplit information
    for splitNo in range(5):
        r2 = eval(f'results["split{splitNo}_test_r2"][bestIndex]')
        mae = eval(f'-results["split{splitNo}_test_neg_MAE"][bestIndex]')
        rmse = eval(f'-results["split{splitNo}_test_neg_RMSE"][bestIndex]')
        mape = eval(f'-results["split{splitNo}_test_neg_MAPE"][bestIndex]')
        # Append all data to the dataframe
        resDf.loc[len(resDf)] = pd.Series(
            data=[splitNo, r2, mae, rmse, mape],
            index=["SplitNo", "R2", "MAE", "RMSE", "MAPE"],
        )
    print(resDf)
    if SAVERESULTS:
        # Add on to existing data if possible
        savePath = os.path.join(__file__, "../splitData.csv")
        if os.path.isfile(savePath):
            data = pd.read_csv(savePath)
            resDf = pd.concat([data, resDf])
        resDf.to_csv(savePath, index=False)
else:
    # Access all info
    log(Back.GREEN, "ALL RESULTS", "\n", start="\n", end="\n")
    PrettyPrinter().pprint(gridSearch.cv_results_)
