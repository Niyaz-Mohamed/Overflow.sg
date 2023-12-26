from data_gen import constructDataset
from modeling import splitDataset

# Import AI related modules
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from sklearn.metrics import (
    r2_score,
)

MODEL = "XGB"
DATERANGE = ["2023-11-26", "2023-11-30"]
PREDTIME = 1

# Get parameter grid
paramGrid = {
    # XGB parameters
    "XGB": {
        "objective": "reg:squarederror",
        "eval_metric": ["rmse", "mae", "logloss"],
        "n_estimators": 100,
        "max_depth": 8,
        "learning_rate": 0.1,
        "subsample": 0.7,
        "gamma": 2,
        "scale_pos_weight": 0,
        "alpha": 10,
        "lambda": 0,
    }
}

# Construct complete dataset
data = constructDataset(
    predictionTime=PREDTIME, restrictDate=DATERANGE, restrictDistance=3.5
)
xTrain, xTest, yTrain, yTest = splitDataset(data)

# Convergence for XGB
if MODEL == "XGB":
    # Prepare training and testing matrices
    dtrain = xgb.DMatrix(xTrain, label=yTrain)
    dtest = xgb.DMatrix(xTest, label=yTest)
    params = paramGrid["XGB"]
    numRounds = params["n_estimators"]
    del params["n_estimators"]

    # Custom evaluation function for R2
    def r2_eval(preds, dmatrix):
        labels = dmatrix.get_label()
        return "r2", r2_score(labels, preds)

    # Train the model
    evals_result = {}
    model = xgb.train(
        paramGrid["XGB"],
        dtrain,
        num_boost_round=numRounds,
        evals=[(dtest, "eval")],
        evals_result=evals_result,
        feval=r2_eval,
    )

    # Access the evaluation metrics over the training period
    print("Evaluation results:")
    for eval_name, eval_dict in evals_result.items():
        print(f"\n{eval_name}:")
        for metric_name, metric_values in eval_dict.items():
            print(f"{metric_name}: {metric_values}")
elif MODEL == "MLP":
    pass
