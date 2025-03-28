import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def fetch_stock_data(ticker, start_date=None, end_date=None, period=None, interval='1d'):
    """
    Fetch stock price data using yfinance.
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str, optional): Start date in 'YYYY-MM-DD' format
        end_date (str, optional): End date in 'YYYY-MM-DD' format
        period (str, optional): Period to fetch (e.g., '1y', '6mo', '5y')
        interval (str): Data interval ('1d', '1wk', '1mo', etc.)
    
    Returns:
        pd.DataFrame: DataFrame containing stock price data
    """
    # Set default dates if not provided
    if start_date is None and period is None:
        period = '5y'  # Default to 5 years if neither start_date nor period is specified
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Fetch data
    if period is not None:
        data = yf.download(ticker, period=period, interval=interval)
    else:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    
    if data.empty:
        raise ValueError(f"No data found for ticker {ticker}")
    
    return data

def calculate_returns(prices, method='log'):
    """
    Calculate returns from price data.
    
    Args:
        prices (pd.Series or array-like): Asset prices
        method (str): Return calculation method ('log' or 'simple')
    
    Returns:
        pd.Series or np.ndarray: Returns series
    """
    if method.lower() == 'log':
        returns = np.diff(np.log(prices))
    elif method.lower() == 'simple':
        returns = np.diff(prices) / prices[:-1]
    else:
        raise ValueError(f"Unknown return calculation method: {method}")
    
    if isinstance(prices, pd.Series):
        index = prices.index[1:]
        return pd.Series(returns, index=index)
    else:
        return returns

def fetch_interest_rate_data(ticker='DGS10', start_date=None, end_date=None, source='fred'):
    """
    Fetch interest rate data (requires fredapi package for 'fred' source).
    
    Args:
        ticker (str): Interest rate ticker/series ID
        start_date (str, optional): Start date in 'YYYY-MM-DD' format
        end_date (str, optional): End date in 'YYYY-MM-DD' format
        source (str): Data source ('fred' or 'yfinance')
    
    Returns:
        pd.Series: Interest rate data
    """
    if source.lower() == 'fred':
        try:
            from fredapi import Fred
        except ImportError:
            raise ImportError("fredapi package required for FRED data. Install with 'pip install fredapi'")
        
        try:
            fred_key = open('fred_api_key.txt').read().strip()
        except FileNotFoundError:
            raise FileNotFoundError("FRED API key required. Create file 'fred_api_key.txt' with your API key.")
        
        fred = Fred(api_key=fred_key)
        
        if start_date is None:
            start_date = '2000-01-01'
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        data = fred.get_series(ticker, start_date, end_date)
        
    elif source.lower() == 'yfinance':
        if start_date is None:
            start_date = '2000-01-01'
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        data = yf.download(f'^{ticker}', start=start_date, end=end_date)['Adj Close']
        
    else:
        raise ValueError(f"Unknown data source: {source}")
    
    # Handle missing values
    if isinstance(data, pd.Series):
        data = data.fillna(method='ffill')
    
    return data

def fetch_vix_data(start_date=None, end_date=None):
    """
    Fetch VIX (volatility index) data.
    
    Args:
        start_date (str, optional): Start date in 'YYYY-MM-DD' format
        end_date (str, optional): End date in 'YYYY-MM-DD' format
    
    Returns:
        pd.Series: VIX data
    """
    # Set default dates if not provided
    if start_date is None:
        start_date = '2000-01-01'
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Fetch VIX data
    vix_data = yf.download('^VIX', start=start_date, end=end_date)
    
    return vix_data['Adj Close']

def create_correlated_returns(n_assets, n_obs, correlation_matrix=None, means=None, volatilities=None, 
                              random_state=None):
    """
    Create synthetic correlated asset returns for testing.
    
    Args:
        n_assets (int): Number of assets
        n_obs (int): Number of observations
        correlation_matrix (np.ndarray, optional): Correlation matrix (n_assets x n_assets)
        means (np.ndarray, optional): Mean returns for each asset
        volatilities (np.ndarray, optional): Volatilities for each asset
        random_state (int, optional): Seed for random number generator
    
    Returns:
        np.ndarray: Correlated returns matrix (n_obs x n_assets)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Set default parameters if not provided
    if correlation_matrix is None:
        # Default: all correlations = 0.5, except diagonal = 1
        correlation_matrix = 0.5 * np.ones((n_assets, n_assets))
        np.fill_diagonal(correlation_matrix, 1.0)
    
    if means is None:
        means = 0.0001 * np.ones(n_assets)  # Default: all means = 0.01% daily
    
    if volatilities is None:
        volatilities = 0.01 * np.ones(n_assets)  # Default: all volatilities = 1% daily
    
    # Create covariance matrix
    vols_matrix = np.diag(volatilities)
    covariance_matrix = vols_matrix @ correlation_matrix @ vols_matrix
    
    # Generate multivariate normal returns
    returns = np.random.multivariate_normal(means, covariance_matrix, size=n_obs)
    
    return returns 