import json
from mlflow.tracking import MlflowClient
import mlflow


# Initialize DagsHub for MLflow tracking

# Set the MLflow experiment
mlflow.set_experiment("Trading_AI_Model")

# Set MLflow tracking URI for DagsHub
mlflow.set_tracking_uri("http://localhost:5000") 

# Path to the JSON file containing run ID and model name
metadata_path = "models/model_metadata.json"

try:
    # Load the run ID and model name from JSON
    with open(metadata_path, 'r') as file:
        model_info = json.load(file)

    run_id = model_info.get('run_id', None)
    model_name = model_info.get('model_name', None)

    if not run_id or not model_name:
        raise ValueError("Run ID or Model Name not found in JSON file.")

    # Create an MLflow client
    client = MlflowClient()

    # Model URI for MLflow
    model_uri = f"runs:/{run_id}/artifacts/{model_name}"

    # Register the model in MLflow
    reg_model = mlflow.register_model(model_uri, model_name)

    # Fetch the model version
    model_version = reg_model.version

    # Transition the model to "Staging"
    new_stage = "Staging"

    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage=new_stage,
        archive_existing_versions=True
    )

    print(f"Model '{model_name}' version {model_version} transitioned to '{new_stage}' stage.")

except FileNotFoundError:
    print(f"Error: Metadata file '{metadata_path}' not found. Ensure training is completed before registering the model.")
except Exception as e:
    print(f"Error during model registration: {str(e)}")
