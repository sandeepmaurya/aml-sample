# MLOps Sample

## References:

https://learn.microsoft.com/en-us/azure/machine-learning/tutorial-azure-ml-in-a-day

https://microsoftlearning.github.io/mslearn-mlops/


## Infra Provisioning
For this sample, we've already provisioned:
* DataStore (diabetes)
  * dev_1_0_0 - Has files for experimenting and creating new models
  * test_1_0_0 - Has files for testing new models. This dataset MUST NOT be used to create models
* Environment (diabetes_1_0_1)
  * This has all dependencies to train the model and run inferencing. See inferencing_conda.yml for dependencies 
* Compute Cluster (smws001cluster)

The dev, stage and prod Deployments are created at runtime if they do not exist.

## Flow
* Member of data science team runs local experiment using the dev dataset
* Once satisfied, she updates train.py. This script returns the new classification model
* She raises a PR for the changes
* PR Build (Jenkinsfile)
  * Runs an experiment using new training script and dev dataset
  * The run, upon completion, records its metrics against the dev dataset
  * The run also outputs the model as an MLFlow model
* Lead data scientist reviews the PR metrics and approves the PR
* PR is merged to main branch that triggers the Main build
* Main Build (Jenkinsfile)
  * Runs an experiment using latest training script and dev dataset
  * Registers the run output model into model registry
  * Deploys the model to Dev env.
  * Tests dev inferencing endpoint against test dataset
  * If the Dev metrics are better than current Prod metrics, deploys the new model to Stage env.
* Lead data scientist can manually validate Stage endpoint and trigger the Prod build
* Prod build (Jenkinsfile-Prod-Deploy)
  * Deploys the input model to Prod env.
