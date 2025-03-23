import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

class StockDataFetcher:
    """Class to fetch stock data from Yahoo Finance"""
    
    def __init__(self, ticker: str):
        """
        Initialize the StockDataFetcher object.
        
        :param ticker: Stock symbol (e.g., 'AAPL' for Apple)
        """
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.data_dir = "data/raw"  # Directory to save data
        os.makedirs(self.data_dir, exist_ok=True)  # Ensure directory exists
        self.filename = os.path.join(self.data_dir, f"{ticker}_stock_data.csv")

    def get_latest_data(self) -> pd.DataFrame:
        """
        Fetch stock data for the last 1 year from today.
        
        :return: Pandas DataFrame containing stock data
        """
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')

        try:
            data = self.stock.history(start=start_date, end=end_date)
            if data.empty:
                print("No data found. Check the ticker symbol or date range.")
                return None
            return data.reset_index()  # Reset index for better formatting
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return None

    def update_csv(self, new_data: pd.DataFrame):
        """
        Update the CSV file by removing the oldest day's data and keeping only the latest 1 year.
        
        :param new_data: DataFrame with the latest stock data
        """
        if new_data is None:
            return

        if os.path.exists(self.filename):
            existing_data = pd.read_csv(self.filename)
            combined_data = pd.concat([existing_data, new_data]).drop_duplicates(subset=['Date'], keep='last')
            combined_data = combined_data.tail(365)  # Keep only last 365 days
        else:
            combined_data = new_data.tail(365)  # If file doesn't exist, take last 365 days only

        combined_data.to_csv(self.filename, index=False)
        print(f"Updated {self.filename} with the latest stock data.")

# Example usage:
if __name__ == "__main__":
    ticker_symbol = "AAPL"  # Change this to any stock ticker you need

    stock_fetcher = StockDataFetcher(ticker_symbol)

    # Fetch the latest stock data
    latest_data = stock_fetcher.get_latest_data()

    # Update the CSV file while keeping only the last 1 year of data
    stock_fetcher.update_csv(latest_data)
