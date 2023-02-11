import os
from time import sleep

import numpy as np
from azure.ai.ml import MLClient, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

import utils

subscription_id = '3ccb9182-11da-487f-9b4f-be7e2fcfd5d3'
resource_group = 'aml'
workspace_name = 'smws001'
experiment_name = 'diabetes_classification'

tenant_id = '305f1b09-dfce-4875-8ae1-287c94373798'
client_id = '7e35756d-5c6b-479e-8d7c-97b15c751b8c'
client_secret = os.environ['CLIENT_SECRET']

auth = ServicePrincipalAuthentication(tenant_id, client_id, client_secret)

ml_client = MLClient(
    auth, subscription_id, resource_group, workspace_name
)
workspace = Workspace(subscription_id, resource_group, workspace_name, auth)

# Run Experiment.
print('Running new experiment...')
command_job = command(
    code='.',
    command='python diabetes_classification/evaluate.py --data_path ${{inputs.data_path}}',
    inputs={
        'data_path': 'dev_1_0_0',
    },
    environment='diabetes_1_0_1@latest',
    environment_variables={'CLIENT_SECRET': client_secret},
    compute='smws001cluster',
    experiment_name=experiment_name,
    description='Train a scikit-learn LogisticRegression on the diabetes dataset.'
)

experiment_job = ml_client.jobs.create_or_update(command_job)
print(f'Studio endpoint: {experiment_job.services["Studio"].endpoint}')
print(f'Started run: {experiment_job.name}')

# Wait for the experiment run to finish.
sleep(30)
run_status = 'None'
run = None
while run_status not in ('Completed', 'Failed', 'Canceled'):
    # Fetch the runs every time we want to check status. Otherwise, status is not refreshed.
    runs = workspace.experiments.get(experiment_name).get_runs()
    run = next(r for r in runs if r.get_details()['runId'] == experiment_job.name)
    run_status = run.status
    print(f'Run Status: {run_status}')
    if run_status not in ('Completed', 'Failed', 'Canceled'):
        sleep(15)
print(f'Run Status: {run_status}')

# Register the output model as a new candidate in registry.
print('\nRegistering run output in registry...')
job_name = experiment_job.name

run_model = Model(
    path=f"azureml://jobs/{job_name}/outputs/artifacts/paths/model/",
    name="diabetes_classification",
    description="Model created from run.",
    type=AssetTypes.MLFLOW_MODEL,
)

registered_model = ml_client.models.create_or_update(run_model)
print(f'Registered model:\n{registered_model}')

# Deploy to dev env.
print('\nDeploying model to dev env. ...')
dev_endpoint_name = 'diabetes-classification-dev'
utils.deploy_model(ml_client, client_secret, 'dev', dev_endpoint_name, registered_model)

# Inference against the test data.
print('\nInferencing against test data on dev deployment...')
df = utils.load_data('test_1_0_0')
X, y = df[['Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure', 'TricepsThickness', 'SerumInsulin', 'BMI',
           'DiabetesPedigree', 'Age']].values, df['Diabetic'].values
scoring_uri = ml_client.online_endpoints.get(dev_endpoint_name).scoring_uri
primary_key = ml_client.online_endpoints.get_keys(dev_endpoint_name).primary_key
response_json = utils.invoke_endpoint(url=scoring_uri, api_key=primary_key, data=X)
y_pred = np.array(response_json)
f1 = f1_score(y, y_pred)
print(f'f1_score: {f1}')
print(f'\nclassification report:\n{classification_report(y, y_pred)}')

# Update model tags with metrics from test data.
registered_model.tags = {"f1_score": f1}
ml_client.models.create_or_update(registered_model)

# Deploy to stage if prod deployment does not exist
# OR Current model is better than prod model
stage_endpoint_name = 'diabetes-classification-stage'
prod_endpoint_name = 'diabetes-classification-prod'
is_prod_endpoint_exists = utils.is_endpoint_exists(ml_client, prod_endpoint_name)
if is_prod_endpoint_exists:
    print('Comparing current model f1 with prod model...')
    prod_deployment = ml_client.online_deployments.get(name='default', endpoint_name=prod_endpoint_name)
    prod_f1 = np.float64(prod_deployment.tags['f1_score'])
    print(f'Prod f1: {prod_f1}, current f1: {f1}')
    if f1 > prod_f1:
        print(f'Current f1 is better than prod. Deploying current model to stage...')
        utils.deploy_model(ml_client, client_secret, 'stage', stage_endpoint_name, registered_model)
    else:
        print(f'Current f1 is not better than prod. Skipped stage deployment.')
else:
    print(f'Prod deployment not found. Deploying current model to stage...')
    utils.deploy_model(ml_client, client_secret, 'stage', stage_endpoint_name, registered_model)
