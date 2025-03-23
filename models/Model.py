import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Load Data
class StockData:
    def __init__(self, filename):
        self.filename = filename
        self.data = None
        self.scaler = MinMaxScaler(feature_range=(0,1))
    
    def load_data(self):
        """Load stock data from CSV and process it."""
        df = pd.read_csv(self.filename, parse_dates=['Date'], index_col='Date')
        self.data = df[['Close']].values  # Only use 'Close' prices
    
    def preprocess_data(self, seq_length=50):
        """Scale data and create sequences for LSTM."""
        self.data = self.scaler.fit_transform(self.data)
        
        X, Y = [], []
        for i in range(len(self.data) - seq_length):
            X.append(self.data[i:i+seq_length])
            Y.append(self.data[i+seq_length])
        
        X, Y = np.array(X), np.array(Y)
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

# Train the Model
def train_model(model, train_loader, epochs=50, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

# Predict & Plot Results
def predict_and_plot(model, X_test, Y_test, scaler):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy()

    # Inverse transform to get original values
    Y_test = scaler.inverse_transform(Y_test.numpy())
    predictions = scaler.inverse_transform(predictions)

    # Plot Results
    plt.figure(figsize=(12,6))
    plt.plot(Y_test, label="Actual Prices", color='blue')
    plt.plot(predictions, label="Predicted Prices", color='red')
    plt.legend()
    plt.title("Stock Price Prediction using LSTM")
    plt.show()

# Main Execution
if __name__ == "__main__":
    # Load Data
    ticker = "AAPL"  # Change to any stock
    data_file = f"{ticker}_stock_data.csv"
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
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize & Train LSTM Model
    model = StockLSTM()
    train_model(model, train_loader)

    # Predict & Plot Results
    predict_and_plot(model, X_test, Y_test, scaler)
