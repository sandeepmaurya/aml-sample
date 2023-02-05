import os
from time import sleep

from azure.ai.ml import MLClient, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import ManagedOnlineDeployment, CodeConfiguration, Environment
from azure.ai.ml.entities import ManagedOnlineEndpoint
from azure.ai.ml.entities import Model
from azureml.core import Workspace
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
workspace = Workspace(subscription_id, resource_group, workspace_name, auth)

# Run Experiment.
print('Running new experiment...')
command_job = command(
    code='.',
    command='python src/diabetes_classification/evaluate.py --data_path ${{inputs.data_path}}',
    inputs={
        'data_path': 'test_1_0_0',
    },
    environment='diabetes_1_0_0@latest',
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


def wait_for_completion(poller):
    while poller.status() not in ('Succeeded', 'Failed'):
        print(f'Status: {poller.status()}')
        sleep(10)
    print(f'Status: {poller.status()}')


# Create dev endpoint if needed.
print('\nCreating dev endpoint if not exists...')
dev_endpoint_name = 'diabetes-classification-dev'
dev_endpoint = None
try:
    dev_endpoint = ml_client.online_endpoints.get(dev_endpoint_name)
except:
    dev_endpoint = ManagedOnlineEndpoint(
        name=dev_endpoint_name,
        description="Diabetes classification - Dev endpoint.",
        auth_mode="key",
        tags={"env": "dev"},
    )
    poller = ml_client.online_endpoints.begin_create_or_update(dev_endpoint)
    wait_for_completion(poller)

# Refresh the dev endpoint with newly created model.
# The inferencing conda env. has few more dependencies than the one we used for training.
print('\nRefreshing default dev deployment...')
default_deployment = ManagedOnlineDeployment(
    name='default',
    endpoint_name=dev_endpoint_name,
    model=registered_model,
    environment=Environment(
        image='mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest',
        conda_file='src/diabetes_classification/inferencing_conda.yml'),
    code_configuration=CodeConfiguration(
        code="src/diabetes_classification", scoring_script="score.py"
    ),
    instance_type="Standard_E2s_v3",
    instance_count=1,
)
poller = ml_client.online_deployments.begin_create_or_update(default_deployment)
wait_for_completion(poller)

# Route all inferencing traffic to this deployment.
print('\nRouting all traffic to default dev deployment...')
dev_endpoint.traffic = {"default": 100}
poller = ml_client.online_endpoints.begin_create_or_update(dev_endpoint)
wait_for_completion(poller)

# Run inferencing against test data

# If current model accuracy > prod accuracy, deploy to Stage endpoint
