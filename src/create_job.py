from azure.ai.ml import MLClient, command
from azure.identity import DefaultAzureCredential

subscription_id = '3ccb9182-11da-487f-9b4f-be7e2fcfd5d3'
resource_group = 'aml'
workspace_name = 'smws001'
experiment_name = 'sm_diabetes_improvements'

ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace_name
)

command_job = command(
    code='./model',
    command='python train.py --data_store_name ${{inputs.data_store_name}} --data_path ${{inputs.data_path}} --reg_rate ${{inputs.reg_rate}}            ',
    inputs={
        'data_store_name': 'diabetes',
        'data_path': '1_0_0',
        'reg_rate': 0.01
    },
    environment='sm_py3_9_13_sklearn1_0_2@latest',
    environment_variables={'CLIENT_SECRET': 'TODO'},
    compute='smws001cluster',
    experiment_name=experiment_name,
    description='Train a scikit-learn LogisticRegression on the diabetes dataset.'
)

returned_job = ml_client.jobs.create_or_update(command_job)
returned_job
