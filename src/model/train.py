# Import libraries

import argparse

import mlflow
from azureml.core import Dataset, Datastore, Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

subscription_id = '3ccb9182-11da-487f-9b4f-be7e2fcfd5d3'
resource_group = 'aml'
workspace_name = 'smws001'
experiment_name = 'sm_diabetes_improvements'

# Output of: az ml workspace show --query mlflow_tracking_uri -g aml -n smws001
mlflow_tracking_uri = 'azureml://centralindia.api.azureml.ms/mlflow/v1.0/subscriptions/3ccb9182-11da-487f-9b4f-be7e2fcfd5d3/resourceGroups/aml/providers/Microsoft.MachineLearningServices/workspaces/smws001'
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment(experiment_name)
mlflow.autolog()
workspace = Workspace(subscription_id, resource_group, workspace_name,
                      auth=ServicePrincipalAuthentication(tenant_id='305f1b09-dfce-4875-8ae1-287c94373798',
                                                          service_principal_id='7e35756d-5c6b-479e-8d7c-97b15c751b8c',
                                                          service_principal_password='TODO'))


def main(args):
    # read data
    df = load_diabetes(args.data_store_name, args.data_path)

    # split data
    X_train, X_test, y_train, y_test = split_data(df)

    # train model
    train_model(args.reg_rate, X_train, X_test, y_train, y_test)


def load_diabetes(data_store_name: str, data_path: str):
    datastore = Datastore.get(workspace, data_store_name)
    dataset = Dataset.Tabular.from_delimited_files(path=(datastore, data_path))
    return dataset.to_pandas_dataframe()


def split_data(df):
    X, y = df[['Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure', 'TricepsThickness', 'SerumInsulin', 'BMI',
               'DiabetesPedigree', 'Age']].values, df['Diabetic'].values
    return train_test_split(X, y, test_size=0.30, random_state=42)


def train_model(reg_rate, X_train, X_test, y_train, y_test):
    LogisticRegression(C=1 / reg_rate, solver="liblinear").fit(X_train, y_train)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_store_name", dest='data_store_name', type=str)
    parser.add_argument("--data_path", dest='data_path', type=str)
    parser.add_argument("--reg_rate", dest='reg_rate', type=float, default=0.01)
    return parser.parse_args()


# run script
# Sample args: --data_store_name diabetes --data_path 1_0_0 --reg_rate 0.001
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
