from sklearn.linear_model import LogisticRegression


def train_model(X_train, y_train):
    model = LogisticRegression(C=1 / 0.1, solver="liblinear").fit(X_train, y_train)
    return model
