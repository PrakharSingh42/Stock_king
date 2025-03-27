# import torch
# import mlflow.pytorch
# import mlflow
# import os
# import mlflow.pyfunc
# import uvicorn
# import numpy as np
# import pandas as pd
# from fastapi import FastAPI
# from pydantic import BaseModel
# from sklearn.preprocessing import MinMaxScaler

# # Load Environment Variables
# # Set MLflow Tracking URI
# mlflow.set_tracking_uri("https://dagshub.com/PrakharSingh42/Stock_king.mlflow")

# # Load Model from DagsHub
# model_name = "StockLSTM"
# run_id = "91ba4bc427f5421fbf37abc9e6abec9d"  # Replace with the actual run ID from `model_metadata.json`

# model_uri = f"runs:/{run_id}/{model_name}"
# model = mlflow.pyfunc.load_model(model_uri)
# # Initialize FastAPI app
# app = FastAPI()

# # Input Data Schema
# class StockInput(BaseModel):
#     prices: list  # List of last 50 closing prices

# # Load Scaler
# scaler = MinMaxScaler(feature_range=(0,1))

# @app.post("/predict")
# def predict(data: StockInput):
#     """Predict next stock price given last 50 closing prices."""
    
#     # Convert input list to numpy array
#     input_data = np.array(data.prices, dtype=np.float32).reshape(-1, 1)
    
#     # Scale input data
#     input_scaled = scaler.fit_transform(input_data)  # Assuming the same scaler as training
    
#     # Convert to tensor
#     input_tensor = torch.tensor(input_scaled.reshape(1, 50, 1), dtype=torch.float32)
    
#     # Move to GPU if available
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     input_tensor = input_tensor.to(device)

#     # Predict
#     model.eval()
#     with torch.no_grad():
#         prediction = model(input_tensor).cpu().numpy()

#     # Inverse transform
#     predicted_price = scaler.inverse_transform(prediction.reshape(-1, 1))[0][0]
    
#     return {"predicted_price": predicted_price}

# # Run FastAPI server
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

import torch
import mlflow.pyfunc
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import os
import json

# Initialize FastAPI
app = FastAPI()

# Load model metadata
metadata_file = "models/model_metadata.json"
if not os.path.exists(metadata_file):
    raise FileNotFoundError("Model metadata file not found!")

with open(metadata_file, "r") as f:
    metadata = json.load(f)

run_id = metadata["run_id"]
model_name = metadata["model_name"]

# Set MLflow tracking URI (DagsHub)
mlflow.set_tracking_uri("https://dagshub.com/PrakharSingh42/Stock_king.mlflow")

# Load the model from MLflow
model_uri = f"runs:/{run_id}/model"
model = mlflow.pytorch.load_model(model_uri)

# Define Input Schema
class ModelInput(BaseModel):
    features: list[float]

@app.post("/predict")
async def predict(data: ModelInput):
    # Convert input list to tensor
    input_tensor = torch.tensor(data.features, dtype=torch.float32).view(1, -1, 1)

    # Ensure model is in evaluation mode
    model.eval()

    # Move tensor to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)
    model.to(device)

    # Perform prediction
    with torch.no_grad():
        prediction = model(input_tensor).cpu().numpy().tolist()

    return {"prediction": prediction}

@app.get("/")
def home():
    return {"message": "Stock Prediction Model is running on FastAPI!"}
