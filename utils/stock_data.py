import yfinance as yf
import pandas as pd
import os
from datetime import datetime

def fetch_option_data(ticker, expiry_date, low_strike, high_strike):
    """
    Fetch option chain data for a given ticker and expiry from Yahoo Finance,
    filtering strikes between low_strike and high_strike.

    Returns filtered DataFrame with relevant option info.
    """
    stock = yf.Ticker(ticker)
    if expiry_date not in stock.options:
        raise ValueError(f"Expiry date {expiry_date} not available for {ticker}")

    calls = stock.option_chain(expiry_date).calls
    calls_filtered = calls[(calls['strike'] >= low_strike) & (calls['strike'] <= high_strike)].copy()

    calls_filtered['expiry'] = expiry_date
    calls_filtered['fetch_date'] = datetime.now().strftime('%Y-%m-%d')

    return calls_filtered[['contractSymbol', 'strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'expiry', 'fetch_date']]

def save_option_data(df, directory, ticker, expiry_date):
    """Save option data DataFrame as CSV."""
    os.makedirs(directory, exist_ok=True)
    filename = f"{ticker}_options_{expiry_date}.csv"
    filepath = os.path.join(directory, filename)
    df.to_csv(filepath, index=False)
    print(f"Saved option data to {filepath}")
    return filepath

    fetch_and_save_all_options(ticker, directory, low_strike, high_strike, low_maturity, high_maturity)
