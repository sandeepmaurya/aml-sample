import mlflow
from sklearn.linear_model import LogisticRegression


def train_model(X_train, y_train):
    reg_strength = 0.1
    mlflow.log_param("reg_strength", reg_strength)
    model = LogisticRegression(C=1 / reg_strength, solver="liblinear").fit(X_train, y_train)
    return model
