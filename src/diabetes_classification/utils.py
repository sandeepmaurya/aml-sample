import json
import os
import ssl
import urllib.request

from azureml.core import Dataset, Datastore, Workspace
from azureml.core.authentication import ServicePrincipalAuthentication


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


def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context


def invokeEndpoint(url, api_key, data):
    allowSelfSignedHttps(True)
    inference_request = {'data': data.tolist()}
    body = str.encode(json.dumps(inference_request))
    headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key),
               'azureml-model-deployment': 'default'}
    req = urllib.request.Request(url, body, headers)
    response = urllib.request.urlopen(req)
    return json.load(response)
