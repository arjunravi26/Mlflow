import os

import mlflow
from mlflow import MlflowClient, set_tracking_uri

# Set the DagsHub API token
os.environ["MLFLOW_TRACKING_USERNAME"] = "arjunravi726"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "b19323a7f7f2713a24d2bd632075fddda73388ee"


# Set the tracking URI
remote_url = "https://dagshub.com/arjunravi726/mlflow.mlflow"
set_tracking_uri(remote_url)

client = MlflowClient()
registered_models = client.search_registered_models()
for model in registered_models:
    print(model.name)

 
def assign_alias_to_stage(model_name, stage, alias):
    """
    Assign an alias to the latest version of a registered model within a specified stage.

    :param model_name: The name of the registered model.
    :param stage: The stage of the model version for which the alias is to be assigned. Can be
                "Production", "Staging", "Archived", or "None".
    :param alias: The alias to assign to the model version.
    :return: None
    """
    latest_mv = client.get_model_version_by_alias(name=model_name, alias=alias)
    print(
        f"Assigning alias '{alias}' to version '{latest_mv.version}' of model '{model_name}'.")
    client.set_registered_model_alias(model_name, "champion", latest_mv.version)

    # client.transition_model_version_stage(name=model_name,version="8",stage='Production', archive_existing_versions=True )

assign_alias_to_stage("ElasticnetWineModel", "Production", "best")
model_info = client.get_model_version_by_alias("ElasticnetWineModel", "best")
model_tags = model_info.tags
print(model_tags)

# Get the model version using a model URI
model_uri = f"models:/ElasticnetWineModel@champion"
model = mlflow.sklearn.load_model(model_uri)

print(model)