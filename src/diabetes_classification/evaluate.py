import argparse
import os

import matplotlib.pyplot as plt
import mlflow
from azureml.core import Dataset, Datastore, Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

import train


def load_data(data_path):
    auth = get_service_principal_auth()
    workspace = get_aml_workspace(auth)

    datastore = Datastore.get(workspace, 'diabetes')
    dataset = Dataset.Tabular.from_delimited_files(path=(datastore, data_path))
    return dataset.to_pandas_dataframe()


def get_aml_workspace(auth):
    subscription_id = '3ccb9182-11da-487f-9b4f-be7e2fcfd5d3'
    resource_group = 'aml'
    workspace_name = 'smws001'
    workspace = Workspace(subscription_id, resource_group, workspace_name, auth)
    return workspace


def get_service_principal_auth():
    tenant_id = '305f1b09-dfce-4875-8ae1-287c94373798'
    client_id = '7e35756d-5c6b-479e-8d7c-97b15c751b8c'
    client_secret = os.environ['CLIENT_SECRET']
    auth = ServicePrincipalAuthentication(tenant_id, client_id, client_secret)
    return auth


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", dest='data_path', type=str)
    return parser.parse_args()


def main(args):
    df = load_data(args.data_path)
    X, y = df[['Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure', 'TricepsThickness', 'SerumInsulin', 'BMI',
               'DiabetesPedigree', 'Age']].values, df['Diabetic'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    model = train.train_model(X_train, y_train)

    log_metrics(model, X_test, y_test)
    plot_roc(X_test, model, y_test)


def plot_roc(X_test, model, y_test):
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    fig = plt.figure(figsize=(6, 4))
    # Plot the diagonal 50% line
    plt.plot([0, 1], [0, 1], 'k--')
    # Plot the FPR and TPR achieved by our model
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig("ROC-Curve.png")
    mlflow.log_artifact("ROC-Curve.png")


def log_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(report)
    mlflow.log_metric("Accuracy", acc)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Recall", recall)
    mlflow.log_metric("F1", f1)
    mlflow.log_metric("ROC AUC", auc)


# CLIENT_SECRET=xxxx python3 evaluate.py --data_path 1_0_0
if __name__ == "__main__":
    args = parse_args()
    main(args)
