import mlflow
from sklearn.linear_model import LogisticRegression


def train_model(X_train, y_train):
    model = LogisticRegression(solver="liblinear").fit(X_train, y_train)
    return model
