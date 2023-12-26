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

# Define grid of model hyperparameters
paramGrid = {
    # SVR parameters
    "SVM": {
        "C": [0.9],
        "kernel": ["rbf"],
        "gamma": ["scale"],
    },
    # CART parameters
    "CART": {
        "max_depth": [None],
        "min_samples_split": [200],
        "min_samples_leaf": [60],
        "max_features": [None],
    },
    # RF parameters
    "RF": {
        "n_estimators": [200],
        "max_depth": [None],
        "min_samples_split": [5],
        "min_samples_leaf": [20],
        "max_features": [1.0],
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
        "hidden_layer_sizes": [(100,)],
        "activation": ["tanh"],
        "alpha": [0.001],
        "learning_rate_init": [0.01],
        "solver": ["adam"],
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

# Access best params
print("\nBest parameters:", gridSearch.best_params_)
print("Best score:", gridSearch.best_score_)

# Access best evaluations
print("\nSummary of Best Results\n")
print(f"R2: {gridSearch.cv_results_['mean_test_r2'][gridSearch.best_index_]}")
print(f"MAE: {-gridSearch.cv_results_['mean_test_neg_MAE'][gridSearch.best_index_]}")
print(f"RMSE: {-gridSearch.cv_results_['mean_test_neg_RMSE'][gridSearch.best_index_]}")
print(f"MAPE: {-gridSearch.cv_results_['mean_test_neg_MAPE'][gridSearch.best_index_]}")
print(f"Fit time: {gridSearch.cv_results_['mean_fit_time'][gridSearch.best_index_]}")
print(
    f"Score time: {gridSearch.cv_results_['mean_score_time'][gridSearch.best_index_]}"
)

# Access all info
print("\nAll Results\n")
PrettyPrinter().pprint(gridSearch.cv_results_)
