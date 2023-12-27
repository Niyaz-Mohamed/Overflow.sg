from data_gen import constructDataset
from pprint import PrettyPrinter
import numpy as np

# Import AI related modules
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
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

# Model to optimise, date range to use for data, and whether to exhaustively or randomly search
MODEL = "XGB"
DATERANGE = ["2023-11-26", "2023-11-30"]
PREDTIME = 0.5
RANDOMIZE = False
NEATPRINT = True

# Define grid of model hyperparameters
paramGrid = {
    # SVR parameters
    "SVM": {
        "C": [0.9],
        # "kernel": ["rbf"],
        # "gamma": ["scale"],
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
        # "max_depth": [None],
        "min_samples_split": [5],
        "min_samples_leaf": [20],
        # "max_features": [1.0],
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
def mean_absolute_percentage_error_eps(y_true, y_pred, epsilon=1e-10):
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

# Prepare search (either random or grid)
if RANDOMIZE:
    gridSearch = RandomizedSearchCV(
        estimator=modelGrid[MODEL],
        param_distributions=paramGrid[MODEL],
        cv=5,
        scoring=scoring,
        refit="r2",
        verbose=1,
        n_jobs=-1,  # Use all available CPU cores
    )
else:
    gridSearch = GridSearchCV(
        estimator=modelGrid[MODEL],
        param_grid=paramGrid[MODEL],
        cv=5,
        scoring=scoring,
        refit="r2",
        verbose=1,
        n_jobs=-1,  # Use all available CPU cores
    )

# Get data and fit
data = constructDataset(
    predictionTime=PREDTIME, restrictDate=DATERANGE, restrictDistance=3.5
)
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
gridSearch.fit(x, y)
model = gridSearch.best_estimator_

# Access best params
print("\nBest parameters:", gridSearch.best_params_)
print("Best score:", gridSearch.best_score_)
results = gridSearch.cv_results_
bestIndex = gridSearch.best_index_

if NEATPRINT:
    # Access best evaluations
    print("\nSummary of Best Results\n")
    print(f"R2: {results['mean_test_r2'][bestIndex]}")
    print(f"MAE: {-results['mean_test_neg_MAE'][bestIndex]}")
    print(f"RMSE: {-results['mean_test_neg_RMSE'][bestIndex]}")
    print(f"MAPE: {-results['mean_test_neg_MAPE'][bestIndex]}")
    print(f"Fit time: {results['mean_fit_time'][bestIndex]}")
    print(f"Score time: {results['mean_score_time'][bestIndex]}")

    # Access split information
    for splitNo in range(5):
        r2 = eval(f'results["split{splitNo}_test_r2"][bestIndex]')
        mae = eval(f'-results["split{splitNo}_test_neg_MAE"][bestIndex]')
        rmse = eval(f'-results["split{splitNo}_test_neg_RMSE"][bestIndex]')
        mape = eval(f'-results["split{splitNo}_test_neg_MAPE"][bestIndex]')

        print(f"\nSplit {splitNo+1}\n")
        print(f"R2: {r2}")
        print(f"MAE: {mae}")
        print(f"RMSE: {rmse}")
        print(f"MAPE: {mape}")
else:
    # Access all info
    print("\nAll Results\n")
    PrettyPrinter().pprint(gridSearch.cv_results_)
