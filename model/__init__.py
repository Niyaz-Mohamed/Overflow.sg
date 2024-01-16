from data_gen import constructDataset
import joblib, os


def loadModel(predTime: float):
    """
    Returns the scikit-learn model with the given prediction lead time.
    Prediction lead time is either 0.5, 1, or 2.
    """
    return joblib.load(os.path.join(__file__, f"../XGB-{predTime}.pkl"))


if __name__ == "__main__":
    print("This is not supposed to be run as is")
