"""
Real Market Data Integration and Analysis

This module provides functionality to fetch, process, and analyze real market data
from various sources including Yahoo Finance, FRED, and other financial APIs.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class MarketDataManager:
    """
    Comprehensive market data manager for fetching and processing financial data.
    """
    
    def __init__(self, cache_data=True):
        """
        Initialize market data manager.
        
        Args:
            cache_data (bool): Whether to cache downloaded data
        """
        self.cache_data = cache_data
        self.cached_data = {}
    
    def fetch_stock_data(self, symbols, start_date=None, end_date=None, period='2y'):
        """
        Fetch stock price data from Yahoo Finance.
        
        Args:
            symbols (list): List of stock symbols
            start_date (str, optional): Start date (YYYY-MM-DD)
            end_date (str, optional): End date (YYYY-MM-DD)
            period (str): Period for data ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            
        Returns:
            dict: Dictionary with price data and metadata
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        cache_key = f"stocks_{'_'.join(symbols)}_{start_date}_{end_date}_{period}"
        
        if self.cache_data and cache_key in self.cached_data:
            return self.cached_data[cache_key]
        
        try:
            # Download data
            if start_date and end_date:
                data = yf.download(symbols, start=start_date, end=end_date, progress=False)
            else:
                data = yf.download(symbols, period=period, progress=False)
            
            if len(symbols) == 1:
                # Single symbol - flatten multi-index
                prices = data['Adj Close'].to_frame(symbols[0])
                volumes = data['Volume'].to_frame(symbols[0])
            else:
                # Multiple symbols
                prices = data['Adj Close']
                volumes = data['Volume']
            
            # Calculate returns
            returns = prices.pct_change().dropna()
            
            # Get company info
            info = {}
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info[symbol] = ticker.info
                except:
                    info[symbol] = {'shortName': symbol, 'sector': 'Unknown'}
            
            result = {
                'prices': prices,
                'returns': returns,
                'volumes': volumes,
                'symbols': symbols,
                'info': info,
                'start_date': prices.index[0] if len(prices) > 0 else None,
                'end_date': prices.index[-1] if len(prices) > 0 else None
            }
            
            if self.cache_data:
                self.cached_data[cache_key] = result
            
            return result
            
        except Exception as e:
            print(f"Error fetching data for {symbols}: {str(e)}")
            return None
    
    def fetch_market_indices(self, period='2y'):
        """
        Fetch major market indices data.
        
        Args:
            period (str): Period for data
            
        Returns:
            dict: Market indices data
        """
        indices = {
            'SPY': 'S&P 500',
            'QQQ': 'NASDAQ 100',
            'IWM': 'Russell 2000',
            'VTI': 'Total Stock Market',
            'VEA': 'Developed Markets',
            'VWO': 'Emerging Markets',
            'AGG': 'Aggregate Bonds',
            'GLD': 'Gold',
            'VNQ': 'REITs'
        }
        
        return self.fetch_stock_data(list(indices.keys()), period=period)
    
    def fetch_sector_etfs(self, period='2y'):
        """
        Fetch sector ETF data for analysis.
        
        Args:
            period (str): Period for data
            
        Returns:
            dict: Sector ETF data
        """
        sector_etfs = {
            'XLK': 'Technology',
            'XLF': 'Financial',
            'XLE': 'Energy',
            'XLV': 'Health Care',
            'XLI': 'Industrial',
            'XLY': 'Consumer Discretionary',
            'XLP': 'Consumer Staples',
            'XLU': 'Utilities',
            'XLB': 'Materials',
            'XLRE': 'Real Estate',
            'XLC': 'Communication Services'
        }
        
        return self.fetch_stock_data(list(sector_etfs.keys()), period=period)
    
    def calculate_market_metrics(self, data):
        """
        Calculate comprehensive market metrics.
        
        Args:
            data (dict): Market data from fetch methods
            
        Returns:
            dict: Market metrics and statistics
        """
        prices = data['prices']
        returns = data['returns']
        
        metrics = {}
        
        for symbol in prices.columns:
            symbol_prices = prices[symbol].dropna()
            symbol_returns = returns[symbol].dropna()
            
            if len(symbol_returns) < 10:  # Need minimum data
                continue
            
            # Basic statistics
            annual_return = symbol_returns.mean() * 252
            annual_vol = symbol_returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
            
            # Risk metrics
            var_95 = np.percentile(symbol_returns, 5)
            var_99 = np.percentile(symbol_returns, 1)
            
            # Drawdown analysis
            cumulative = (1 + symbol_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Trend analysis
            recent_return = (symbol_prices.iloc[-1] / symbol_prices.iloc[-22] - 1) if len(symbol_prices) >= 22 else 0
            ytd_return = (symbol_prices.iloc[-1] / symbol_prices.iloc[0] - 1)
            
            # Volatility analysis
            rolling_vol = symbol_returns.rolling(window=21).std() * np.sqrt(252)
            vol_of_vol = rolling_vol.std()
            
            # Momentum indicators
            momentum_1m = recent_return
            momentum_3m = (symbol_prices.iloc[-1] / symbol_prices.iloc[-63] - 1) if len(symbol_prices) >= 63 else 0
            momentum_6m = (symbol_prices.iloc[-1] / symbol_prices.iloc[-126] - 1) if len(symbol_prices) >= 126 else 0
            
            metrics[symbol] = {
                'annual_return': annual_return,
                'annual_volatility': annual_vol,
                'sharpe_ratio': sharpe_ratio,
                'var_95': var_95,
                'var_99': var_99,
                'max_drawdown': max_drawdown,
                'recent_return_1m': momentum_1m,
                'momentum_3m': momentum_3m,
                'momentum_6m': momentum_6m,
                'ytd_return': ytd_return,
                'volatility_of_volatility': vol_of_vol,
                'current_price': symbol_prices.iloc[-1],
                'price_change_pct': symbol_returns.iloc[-1],
                'avg_volume': data['volumes'][symbol].mean() if symbol in data['volumes'].columns else 0
            }
        
        return metrics
    
    def correlation_analysis(self, data):
        """
        Perform correlation analysis on the data.
        
        Args:
            data (dict): Market data
            
        Returns:
            dict: Correlation analysis results
        """
        returns = data['returns']
        
        # Static correlation
        corr_matrix = returns.corr()
        
        # Rolling correlation (if enough data)
        rolling_corr = {}
        if len(returns) >= 63:
            window = 63
            for i, col1 in enumerate(returns.columns):
                for j, col2 in enumerate(returns.columns):
                    if i < j:  # Only upper triangle
                        pair_name = f"{col1}_{col2}"
                        rolling_corr[pair_name] = returns[col1].rolling(window).corr(returns[col2])
        
        # Average correlation (excluding self-correlation)
        n_assets = len(corr_matrix.columns)
        if n_assets > 1:
            avg_correlation = (corr_matrix.sum().sum() - n_assets) / (n_assets * (n_assets - 1))
        else:
            avg_correlation = 0
        
        # Eigenvalue analysis for risk concentration
        eigenvalues = np.linalg.eigvals(corr_matrix.values)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove near-zero eigenvalues
        
        # Participation ratio (measure of diversification)
        participation_ratio = 1 / np.sum((eigenvalues / eigenvalues.sum())**2)
        
        return {
            'correlation_matrix': corr_matrix,
            'rolling_correlations': rolling_corr,
            'average_correlation': avg_correlation,
            'eigenvalues': eigenvalues,
            'participation_ratio': participation_ratio,
            'effective_rank': participation_ratio / len(eigenvalues) if len(eigenvalues) > 0 else 0
        }
    
    def market_stress_indicators(self, data):
        """
        Calculate market stress and systemic risk indicators.
        
        Args:
            data (dict): Market data
            
        Returns:
            dict: Stress indicators
        """
        returns = data['returns']
        
        # VIX proxy (average volatility)
        rolling_vol = returns.rolling(window=21).std() * np.sqrt(252) * 100
        vix_proxy = rolling_vol.mean(axis=1)
        
        # Stress indicator based on extreme correlations
        corr_analysis = self.correlation_analysis(data)
        stress_periods = []
        
        if corr_analysis['rolling_correlations']:
            # Periods when correlations spike (flight to quality)
            for pair, rolling_corr in corr_analysis['rolling_correlations'].items():
                high_corr_periods = rolling_corr > rolling_corr.quantile(0.9)
                stress_periods.append(high_corr_periods)
        
        # Combined stress indicator
        if stress_periods:
            stress_indicator = pd.concat(stress_periods, axis=1).sum(axis=1)
            stress_indicator = stress_indicator / len(stress_periods)  # Normalize
        else:
            stress_indicator = pd.Series(0, index=returns.index)
        
        # Market turbulence (Mahalanobis distance)
        if len(returns.columns) > 1 and len(returns) > len(returns.columns):
            try:
                mean_returns = returns.mean()
                cov_matrix = returns.cov()
                inv_cov = np.linalg.pinv(cov_matrix.values)
                
                turbulence = []
                for date, row in returns.iterrows():
                    diff = row.values - mean_returns.values
                    turb = np.dot(np.dot(diff, inv_cov), diff.T)
                    turbulence.append(turb)
                
                turbulence = pd.Series(turbulence, index=returns.index)
            except:
                turbulence = pd.Series(0, index=returns.index)
        else:
            turbulence = pd.Series(0, index=returns.index)
        
        # Absorption ratio (concentration of risk)
        if len(returns.columns) > 2:
            try:
                # Rolling PCA to measure risk concentration
                window = min(63, len(returns) // 2)
                absorption_ratios = []
                
                for i in range(window, len(returns)):
                    window_returns = returns.iloc[i-window:i]
                    cov_matrix = window_returns.cov()
                    eigenvalues = np.linalg.eigvals(cov_matrix.values)
                    eigenvalues = eigenvalues[eigenvalues > 0]
                    
                    if len(eigenvalues) >= 2:
                        # Fraction of variance explained by first eigenvalue
                        absorption_ratio = eigenvalues[0] / eigenvalues.sum()
                    else:
                        absorption_ratio = 1.0
                    
                    absorption_ratios.append(absorption_ratio)
                
                absorption_ratio_series = pd.Series(absorption_ratios, index=returns.index[window:])
            except:
                absorption_ratio_series = pd.Series(0.5, index=returns.index)
        else:
            absorption_ratio_series = pd.Series(0.5, index=returns.index)
        
        return {
            'vix_proxy': vix_proxy,
            'stress_indicator': stress_indicator,
            'turbulence': turbulence,
            'absorption_ratio': absorption_ratio_series
        }
    
    def generate_market_report(self, symbols=None, period='1y'):
        """
        Generate comprehensive market analysis report.
        
        Args:
            symbols (list, optional): List of symbols to analyze
            period (str): Period for analysis
            
        Returns:
            dict: Comprehensive market report
        """
        if symbols is None:
            # Default portfolio of major assets
            symbols = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT', 'VNQ']
        
        # Fetch data
        print(f"Fetching market data for {symbols}...")
        data = self.fetch_stock_data(symbols, period=period)
        
        if data is None:
            return {'error': 'Failed to fetch market data'}
        
        print("Calculating market metrics...")
        metrics = self.calculate_market_metrics(data)
        
        print("Performing correlation analysis...")
        correlation_analysis = self.correlation_analysis(data)
        
        print("Calculating stress indicators...")
        stress_indicators = self.market_stress_indicators(data)
        
        # Portfolio-level analysis
        returns = data['returns']
        equal_weight_returns = returns.mean(axis=1)
        
        portfolio_metrics = {
            'equal_weight_return': equal_weight_returns.mean() * 252,
            'equal_weight_volatility': equal_weight_returns.std() * np.sqrt(252),
            'equal_weight_sharpe': (equal_weight_returns.mean() / equal_weight_returns.std()) * np.sqrt(252),
            'portfolio_var_95': np.percentile(equal_weight_returns, 5),
            'portfolio_max_drawdown': self._calculate_max_drawdown(equal_weight_returns)
        }
        
        # Current market conditions
        current_conditions = {
            'date': data['end_date'],
            'market_volatility': stress_indicators['vix_proxy'].iloc[-1] if len(stress_indicators['vix_proxy']) > 0 else np.nan,
            'average_correlation': correlation_analysis['average_correlation'],
            'stress_level': stress_indicators['stress_indicator'].iloc[-1] if len(stress_indicators['stress_indicator']) > 0 else 0,
            'risk_concentration': stress_indicators['absorption_ratio'].iloc[-1] if len(stress_indicators['absorption_ratio']) > 0 else 0.5
        }
        
        return {
            'data': data,
            'individual_metrics': metrics,
            'correlation_analysis': correlation_analysis,
            'stress_indicators': stress_indicators,
            'portfolio_metrics': portfolio_metrics,
            'current_conditions': current_conditions,
            'period': period,
            'symbols': symbols
        }
    
    def _calculate_max_drawdown(self, returns):
        """Helper function to calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def get_sp500_constituents(self, limit=50):
        """
        Get S&P 500 constituent symbols (simplified version).
        
        Args:
            limit (int): Limit number of symbols returned
            
        Returns:
            list: List of S&P 500 symbols
        """
        # This is a simplified list of major S&P 500 stocks
        # In practice, you'd fetch this from a reliable source
        sp500_major = [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'GOOG', 'BRK-B', 'UNH', 'JNJ', 'META',
            'NVDA', 'JPM', 'V', 'PG', 'HD', 'CVX', 'MA', 'BAC', 'ABBV', 'PFE',
            'AVGO', 'KO', 'LLY', 'TMO', 'WMT', 'COST', 'MRK', 'DIS', 'ABT', 'ACN',
            'VZ', 'ADBE', 'DHR', 'CRM', 'NFLX', 'XOM', 'NKE', 'TXN', 'QCOM', 'NEE',
            'RTX', 'BMY', 'UPS', 'PM', 'ORCL', 'HON', 'LIN', 'COP', 'SBUX', 'LOW'
        ]
        
        return sp500_major[:limit]