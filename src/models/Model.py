import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import mlflow
import mlflow.pytorch
import json
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import dagshub
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise ValueError("DAGSHUB_TOKEN not found!")
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
# Set MLflow tracking URI to DagsHub
mlflow.set_tracking_uri("https://dagshub.com/PrakharSingh42/Stock_king.mlflow")
mlflow.set_experiment("Stock_Prediction")

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Load Data
class StockData:
    def __init__(self, filename):
        self.filename = filename
        self.data = None
        self.scaler = MinMaxScaler(feature_range=(0,1))
    
    def load_data(self):
        """Load stock data from CSV and process it."""
        df = pd.read_csv(self.filename, parse_dates=['Date'], index_col='Date')
        self.data = df[['Close']].values.astype(np.float32)  # Reduce memory usage
    
    def preprocess_data(self, seq_length=50):
        """Scale data and create sequences for LSTM."""
        self.data = self.scaler.fit_transform(self.data)
        
        X, Y = [], []
        for i in range(len(self.data) - seq_length):
            X.append(self.data[i:i+seq_length])
            Y.append(self.data[i+seq_length])
        
        X, Y = np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)  # Convert to float32
        return X, Y, self.scaler

# Define LSTM Model
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Get the last output

# Train the Model with MLflow
def train_model(model, train_loader, epochs=50, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    with mlflow.start_run() as run:
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("hidden_size", model.lstm.hidden_size)
        mlflow.log_param("num_layers", model.lstm.num_layers)
        
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, Y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            mlflow.log_metric("loss", avg_loss, step=epoch+1)
            
            if (epoch+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
        
        # Save trained model
        model_path = "models/model.pkl"
        torch.save(model.state_dict(), model_path)
        mlflow.pytorch.log_model(model, "model")
        print("Model saved and logged to DagsHub!")

        # Save run ID and model name to JSON
        run_id = run.info.run_id
        model_name = "StockLSTM"
        metadata = {
            "run_id": run_id,
            "model_name": model_name
        }
        with open("models/model_metadata.json", "w") as json_file:
            json.dump(metadata, json_file)
        print(f"Run ID and model name saved to models/model_metadata.json")

# Predict & Plot Results
def predict_and_plot(model, X_test, Y_test, scaler):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).detach().cpu().numpy()  # Reduce memory usage
    
    # Inverse transform to get original values
    Y_test = scaler.inverse_transform(Y_test.detach().cpu().numpy())  # Free memory
    predictions = scaler.inverse_transform(predictions)

# Main Execution
if __name__ == "__main__":
    # Load Data
    ticker = "AAPL"  # Change to any stock
    data_file = f"data/processed/{ticker}_processed_data.csv"
    seq_length = 50

    stock_data = StockData(data_file)
    stock_data.load_data()
    X, Y, scaler = stock_data.preprocess_data(seq_length)

    # Split Data into Train & Test
    train_size = int(len(X) * 0.8)
    X_train, Y_train = X[:train_size], Y[:train_size]
    X_test, Y_test = X[train_size:], Y[train_size:]

    # Convert to Tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32)

    # DataLoader
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Reduce batch size to save memory

    # Initialize & Train LSTM Model
    model = StockLSTM()
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Move tensors to the same device
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_test, Y_test = X_test.to(device), Y_test.to(device)

    train_model(model, train_loader)

    # Predict & Plot Results
    predict_and_plot(model, X_test, Y_test, scaler)

    # Free up memory
    del X_train, Y_train, X_test, Y_test, train_loader, model
    torch.cuda.empty_cache()