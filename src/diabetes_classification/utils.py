import json
import os
import ssl
import urllib.request
from time import sleep

from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, CodeConfiguration
from azure.core.exceptions import ResourceNotFoundError
from azureml.core import Dataset, Datastore, Workspace
from azureml.core.authentication import ServicePrincipalAuthentication

tenant_id = '305f1b09-dfce-4875-8ae1-287c94373798'
client_id = '7e35756d-5c6b-479e-8d7c-97b15c751b8c'
subscription_id = '3ccb9182-11da-487f-9b4f-be7e2fcfd5d3'
resource_group = 'aml'
workspace_name = 'smws001'


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


def get_ml_client():
    auth = get_service_principal_auth()
    return MLClient(
        auth, subscription_id, resource_group, workspace_name
    )


def get_service_principal_auth():
    client_secret = os.environ['CLIENT_SECRET']
    auth = ServicePrincipalAuthentication(tenant_id, client_id, client_secret)
    return auth


def allow_self_signed_https(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context


def wait_for_completion(poller):
    while poller.status() not in ('Succeeded', 'Failed'):
        print(f'Status: {poller.status()}')
        sleep(30)
    print(f'Status: {poller.status()}')


def is_endpoint_exists(ml_client, endpoint_name):
    try:
        ml_client.online_endpoints.get(endpoint_name)
        return True
    except ResourceNotFoundError:
        return False


def deploy_model(ml_client, client_secret, env_name, endpoint_name, model):
    # Create endpoint if not exists.
    endpoint = None
    try:
        endpoint = ml_client.online_endpoints.get(endpoint_name)
    except ResourceNotFoundError:
        print(f'\nCreating {endpoint_name} endpoint...')
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description=endpoint_name,
            auth_mode="key",
            tags={"env": env_name},
        )
        poller = ml_client.online_endpoints.begin_create_or_update(endpoint)
        wait_for_completion(poller)

    # Create default deployment.
    print(f'\nCreating default deployment...')
    default_deployment = ManagedOnlineDeployment(
        name='default',
        endpoint_name=endpoint_name,
        model=model,
        environment='diabetes_1_0_1@latest',
        environment_variables={'CLIENT_SECRET': client_secret},
        code_configuration=CodeConfiguration(
            code="src/diabetes_classification", scoring_script="score.py"
        ),
        instance_type="Standard_F2s_v2",
        instance_count=1,
        tags=model.tags
    )
    poller = ml_client.online_deployments.begin_create_or_update(default_deployment)
    wait_for_completion(poller)

    # Route all inferencing traffic to default deployment.
    print('\nRouting all traffic to default deployment...')
    endpoint.traffic = {"default": 100}
    poller = ml_client.online_endpoints.begin_create_or_update(endpoint)
    wait_for_completion(poller)
    return default_deployment


def invoke_endpoint(url, api_key, data):
    allow_self_signed_https(True)
    inference_request = {'data': data.tolist()}
    body = str.encode(json.dumps(inference_request))
    headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key),
               'azureml-model-deployment': 'default'}
    req = urllib.request.Request(url, body, headers)
    response = urllib.request.urlopen(req)
    return json.load(response)
