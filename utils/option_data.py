import yfinance as yf
import pandas as pd
import os
import json
from datetime import datetime

def fetch_option_data(ticker, expiry_date, low_strike, high_strike):
    """
    Fetch option chain data for a given ticker and expiry from Yahoo Finance,
    filtering strikes between low_strike and high_strike.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. 'AAPL').
    expiry_date : str
        Expiration date in 'YYYY-MM-DD' format.
    low_strike : float
        Minimum strike price to filter options.
    high_strike : float
        Maximum strike price to filter options.

    Returns
    -------
    df : pd.DataFrame
        Filtered DataFrame containing option data.
    """
    stock = yf.Ticker(ticker)
    options_dates = stock.options
    if expiry_date not in options_dates:
        raise ValueError(f"Expiry date {expiry_date} not found in available options: {options_dates}")

    options_chain = stock.option_chain(expiry_date)
    calls = options_chain.calls

    # Filter calls by strike
    calls_filtered = calls[(calls['strike'] >= low_strike) & (calls['strike'] <= high_strike)].copy()

    # Select relevant columns
    df = calls_filtered[['contractSymbol', 'strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest']].copy()

    # Add expiry and fetch date
    df.loc[:, 'expiry'] = expiry_date
    df.loc[:, 'fetch_date'] = datetime.now().strftime('%Y-%m-%d')

    return df

def save_option_data(df, directory, ticker, expiry_date):
    """
    Save option data DataFrame to JSON or CSV for later processing.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of option data.
    directory : str
        Directory path where to save the data.
    ticker : str
        Stock ticker symbol.
    expiry_date : str
        Expiry date string.

    Returns
    -------
    filepath : str
        Path to the saved data file.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = f"{ticker}_options_{expiry_date}.csv"
    filepath = os.path.join(directory, filename)
    df.to_csv(filepath, index=False)
    print(f"Saved option data to {filepath}")
    return filepath

def fetch_and_save_all_options(ticker, directory, low_strike, high_strike, low_maturity, high_maturity):
    stock = yf.Ticker(ticker)
    options_dates = stock.options

    def in_range(date_str):
        d = datetime.strptime(date_str, '%Y-%m-%d')
        return low_maturity <= d <= high_maturity

    valid_expiries = [d for d in options_dates if in_range(d)]

    for expiry in valid_expiries:
        df = fetch_option_data(ticker, expiry, low_strike, high_strike)
        save_option_data(df, directory, ticker, expiry)

if __name__ == "__main__":
    ticker = 'AAPL'
    directory = './option_data'
    low_strike = 140
    high_strike = 180
    low_maturity = datetime.strptime('2025-07-01', '%Y-%m-%d')
    high_maturity = datetime.strptime('2025-09-01', '%Y-%m-%d')

    fetch_and_save_all_options(ticker, directory, low_strike, high_strike, low_maturity, high_maturity)

    # Initialize and run your DupireLocalVolatility model here, using data in directory
