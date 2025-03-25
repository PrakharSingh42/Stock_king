import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

class StockDataPreprocessor:
    """Class for preprocessing stock data"""

    def __init__(self, ticker: str):
        """
        Initialize the StockDataPreprocessor object.

        :param ticker: Stock symbol (e.g., 'AAPL' for Apple)
        """
        self.ticker = ticker
        self.raw_data_path = f"data/raw/{ticker}_stock_data.csv"
        self.processed_data_path = f"data/processed/{ticker}_processed_data.csv"
        os.makedirs("data/processed", exist_ok=True)  # Ensure processed directory exists

    def load_data(self) -> pd.DataFrame:
        """
        Load stock data from the raw CSV file.

        :return: Pandas DataFrame
        """
        if not os.path.exists(self.raw_data_path):
            print(f"Error: File {self.raw_data_path} not found!")
            return None

        df = pd.read_csv(self.raw_data_path)
        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform preprocessing steps on the stock data.

        :param df: Raw stock data DataFrame
        :return: Processed DataFrame
        """
        if df is None or df.empty:
            print("Error: No data available for preprocessing.")
            return None

        # Convert 'Date' column to datetime format
        df['Date'] = pd.to_datetime(df['Date'])

        # Fill missing values using forward fill
        df.fillna(method='ffill', inplace=True)

        # Remove duplicates
        df.drop_duplicates(subset=['Date'], keep='last', inplace=True)

        # Feature Engineering - Adding Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()  # 20-day Simple Moving Average
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()  # 20-day Exponential Moving Average

        # Normalize numerical columns using MinMaxScaler
        scaler = MinMaxScaler()
        df[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume']])

        return df

    def save_processed_data(self, df: pd.DataFrame):
        """
        Save the processed stock data to a new CSV file.

        :param df: Processed DataFrame
        """
        if df is None:
            return

        df.to_csv(self.processed_data_path, index=False)
        print(f"Processed data saved to {self.processed_data_path}")

# Example usage:
if __name__ == "__main__":
    ticker_symbol = "AAPL"  # Change this to any stock ticker you need

    preprocessor = StockDataPreprocessor(ticker_symbol)

    # Load raw data
    raw_data = preprocessor.load_data()

    # Preprocess the data
    processed_data = preprocessor.preprocess_data(raw_data)

    # Save the processed data
    preprocessor.save_processed_data(processed_data)
