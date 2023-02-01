import os

from azure.ai.ml import MLClient, command
from azureml.core.authentication import ServicePrincipalAuthentication

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

command_job = command(
    code='.',
    command='python src/diabetes_classification/evaluate.py --data_path ${{inputs.data_path}}',
    inputs={
        'data_path': '1_0_0',
    },
    environment='diabetes_1_0_0@latest',
    environment_variables={'CLIENT_SECRET': client_secret},
    compute='smws001cluster',
    experiment_name=experiment_name,
    description='Train a scikit-learn LogisticRegression on the diabetes dataset.'
)

returned_job = ml_client.jobs.create_or_update(command_job)
returned_job
