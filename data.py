import pandas as pd
import numpy as np

# --------------------------
# Part 1: Generate Synthetic Stock Price Data
# --------------------------
num_days = 200
start_price = 100.0  # Starting price
np.random.seed(42)   # For reproducibility

# Generate a date range
dates = pd.date_range(start='2020-01-01', periods=num_days, freq='D')

# Initialize lists for the stock data
open_prices = []
high_prices = []
low_prices = []
close_prices = []
volumes = []

price = start_price

for _ in range(num_days):
    # Simulate a daily return from a normal distribution (mean=0, std=0.02)
    daily_return = np.random.normal(0, 0.02)
    
    open_price = price
    # Calculate close price using the daily return
    close_price = open_price * (1 + daily_return)
    
    # High is a bit above the max of open and close, low a bit below the min
    high_price = max(open_price, close_price) * (1 + np.random.uniform(0, 0.01))
    low_price = min(open_price, close_price) * (1 - np.random.uniform(0, 0.01))
    
    volume = np.random.randint(100000, 1000000)
    
    open_prices.append(round(open_price, 2))
    close_prices.append(round(close_price, 2))
    high_prices.append(round(high_price, 2))
    low_prices.append(round(low_price, 2))
    volumes.append(volume)
    
    # Update price for the next day using close price
    price = close_price

# Create a DataFrame
df = pd.DataFrame({
    'Date': dates,
    'Open': open_prices,
    'High': high_prices,
    'Low': low_prices,
    'Close': close_prices,
    'Volume': volumes
})

# Save the DataFrame to a CSV file
csv_filename = 'synthetic_stock_prices.csv'
df.to_csv(csv_filename, index=False)
print(f"Synthetic stock price dataset saved as '{csv_filename}'")
