"""
Black-Scholes Option Pricing Model

This module implements the classic Black-Scholes-Merton option pricing model
with extensions for dividend-paying stocks and American options approximations.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import pandas as pd


class BlackScholesModel:
    """
    Black-Scholes option pricing model with comprehensive analytics.
    """
    
    def __init__(self, spot_price, strike_price, time_to_expiry, risk_free_rate, 
                 volatility, dividend_yield=0.0):
        """
        Initialize Black-Scholes model parameters.
        
        Args:
            spot_price (float): Current price of underlying asset
            strike_price (float): Strike price of option
            time_to_expiry (float): Time to expiration in years
            risk_free_rate (float): Risk-free interest rate (annual)
            volatility (float): Volatility of underlying asset (annual)
            dividend_yield (float): Continuous dividend yield (annual)
        """
        self.S = spot_price
        self.K = strike_price
        self.T = time_to_expiry
        self.r = risk_free_rate
        self.sigma = volatility
        self.q = dividend_yield
        
        # Pre-calculate d1 and d2 for efficiency
        self._calculate_d_parameters()
    
    def _calculate_d_parameters(self):
        """Calculate d1 and d2 parameters used in Black-Scholes formula."""
        if self.T <= 0:
            self.d1 = np.inf if self.S > self.K else -np.inf
            self.d2 = self.d1
        else:
            sqrt_t = np.sqrt(self.T)
            self.d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * sqrt_t)
            self.d2 = self.d1 - self.sigma * sqrt_t
    
    def call_price(self):
        """
        Calculate European call option price.
        
        Returns:
            float: Call option price
        """
        if self.T <= 0:
            return max(self.S - self.K, 0)
        
        return (self.S * np.exp(-self.q * self.T) * norm.cdf(self.d1) - 
                self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2))
    
    def put_price(self):
        """
        Calculate European put option price.
        
        Returns:
            float: Put option price
        """
        if self.T <= 0:
            return max(self.K - self.S, 0)
        
        return (self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) - 
                self.S * np.exp(-self.q * self.T) * norm.cdf(-self.d1))
    
    def delta(self, option_type='call'):
        """
        Calculate option delta (price sensitivity to underlying price).
        
        Args:
            option_type (str): 'call' or 'put'
            
        Returns:
            float: Delta value
        """
        if self.T <= 0:
            if option_type == 'call':
                return 1.0 if self.S > self.K else 0.0
            else:
                return -1.0 if self.S < self.K else 0.0
        
        if option_type == 'call':
            return np.exp(-self.q * self.T) * norm.cdf(self.d1)
        else:
            return -np.exp(-self.q * self.T) * norm.cdf(-self.d1)
    
    def gamma(self):
        """
        Calculate option gamma (delta sensitivity to underlying price).
        
        Returns:
            float: Gamma value
        """
        if self.T <= 0:
            return 0.0
        
        return (np.exp(-self.q * self.T) * norm.pdf(self.d1)) / (self.S * self.sigma * np.sqrt(self.T))
    
    def theta(self, option_type='call'):
        """
        Calculate option theta (time decay).
        
        Args:
            option_type (str): 'call' or 'put'
            
        Returns:
            float: Theta value (per day)
        """
        if self.T <= 0:
            return 0.0
        
        sqrt_t = np.sqrt(self.T)
        
        common_term = -(self.S * norm.pdf(self.d1) * self.sigma * np.exp(-self.q * self.T)) / (2 * sqrt_t)
        
        if option_type == 'call':
            theta = (common_term - 
                    self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2) +
                    self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(self.d1))
        else:
            theta = (common_term + 
                    self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) -
                    self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(-self.d1))
        
        return theta / 365  # Convert to per-day
    
    def vega(self):
        """
        Calculate option vega (volatility sensitivity).
        
        Returns:
            float: Vega value (per 1% volatility change)
        """
        if self.T <= 0:
            return 0.0
        
        return self.S * np.exp(-self.q * self.T) * norm.pdf(self.d1) * np.sqrt(self.T) / 100
    
    def rho(self, option_type='call'):
        """
        Calculate option rho (interest rate sensitivity).
        
        Args:
            option_type (str): 'call' or 'put'
            
        Returns:
            float: Rho value (per 1% interest rate change)
        """
        if self.T <= 0:
            return 0.0
        
        if option_type == 'call':
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2) / 100
        else:
            return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2) / 100
    
    def all_greeks(self, option_type='call'):
        """
        Calculate all Greeks for the option.
        
        Args:
            option_type (str): 'call' or 'put'
            
        Returns:
            dict: Dictionary containing all Greeks
        """
        return {
            'delta': self.delta(option_type),
            'gamma': self.gamma(),
            'theta': self.theta(option_type),
            'vega': self.vega(),
            'rho': self.rho(option_type)
        }
    
    def implied_volatility(self, market_price, option_type='call', max_iterations=100, tolerance=1e-6):
        """
        Calculate implied volatility from market price using Brent's method.
        
        Args:
            market_price (float): Observed market price of option
            option_type (str): 'call' or 'put'
            max_iterations (int): Maximum iterations for convergence
            tolerance (float): Convergence tolerance
            
        Returns:
            float: Implied volatility
        """
        def price_diff(vol):
            temp_model = BlackScholesModel(self.S, self.K, self.T, self.r, vol, self.q)
            if option_type == 'call':
                model_price = temp_model.call_price()
            else:
                model_price = temp_model.put_price()
            return model_price - market_price
        
        try:
            # Use Brent's method to find root
            iv = brentq(price_diff, 0.001, 5.0, maxiter=max_iterations, xtol=tolerance)
            return iv
        except ValueError:
            # If Brent's method fails, return NaN
            return np.nan
    
    def american_option_price(self, option_type='call', n_steps=100):
        """
        Approximate American option price using binomial tree.
        
        Args:
            option_type (str): 'call' or 'put'
            n_steps (int): Number of time steps in binomial tree
            
        Returns:
            float: American option price
        """
        dt = self.T / n_steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((self.r - self.q) * dt) - d) / (u - d)
        
        # Initialize asset prices at maturity
        asset_prices = np.zeros(n_steps + 1)
        for i in range(n_steps + 1):
            asset_prices[i] = self.S * (u ** (n_steps - i)) * (d ** i)
        
        # Initialize option values at maturity
        option_values = np.zeros(n_steps + 1)
        for i in range(n_steps + 1):
            if option_type == 'call':
                option_values[i] = max(asset_prices[i] - self.K, 0)
            else:
                option_values[i] = max(self.K - asset_prices[i], 0)
        
        # Work backwards through the tree
        for step in range(n_steps - 1, -1, -1):
            for i in range(step + 1):
                # Update asset price
                asset_price = self.S * (u ** (step - i)) * (d ** i)
                
                # Calculate continuation value
                continuation_value = np.exp(-self.r * dt) * (p * option_values[i] + (1 - p) * option_values[i + 1])
                
                # Calculate exercise value
                if option_type == 'call':
                    exercise_value = max(asset_price - self.K, 0)
                else:
                    exercise_value = max(self.K - asset_price, 0)
                
                # Choose maximum of continuation and exercise
                option_values[i] = max(continuation_value, exercise_value)
        
        return option_values[0]
    
    def option_chain(self, strike_range, option_type='both'):
        """
        Generate option chain for a range of strike prices.
        
        Args:
            strike_range (list or np.array): Range of strike prices
            option_type (str): 'call', 'put', or 'both'
            
        Returns:
            pd.DataFrame: Option chain with prices and Greeks
        """
        results = []
        
        for strike in strike_range:
            temp_model = BlackScholesModel(self.S, strike, self.T, self.r, self.sigma, self.q)
            
            row_data = {
                'strike': strike,
                'moneyness': self.S / strike
            }
            
            if option_type in ['call', 'both']:
                call_price = temp_model.call_price()
                call_greeks = temp_model.all_greeks('call')
                row_data.update({
                    'call_price': call_price,
                    'call_delta': call_greeks['delta'],
                    'call_gamma': call_greeks['gamma'],
                    'call_theta': call_greeks['theta'],
                    'call_vega': call_greeks['vega'],
                    'call_rho': call_greeks['rho']
                })
            
            if option_type in ['put', 'both']:
                put_price = temp_model.put_price()
                put_greeks = temp_model.all_greeks('put')
                row_data.update({
                    'put_price': put_price,
                    'put_delta': put_greeks['delta'],
                    'put_gamma': put_greeks['gamma'],
                    'put_theta': put_greeks['theta'],
                    'put_vega': put_greeks['vega'],
                    'put_rho': put_greeks['rho']
                })
            
            results.append(row_data)
        
        return pd.DataFrame(results)
    
    def volatility_surface(self, strike_range, time_range, base_volatility=None):
        """
        Generate implied volatility surface using smile/skew models.
        
        Args:
            strike_range (list): Range of strike prices
            time_range (list): Range of times to expiration
            base_volatility (float, optional): Base volatility for smile model
            
        Returns:
            pd.DataFrame: Volatility surface
        """
        if base_volatility is None:
            base_volatility = self.sigma
        
        results = []
        
        for time in time_range:
            for strike in strike_range:
                # Simple volatility smile model (parabolic)
                moneyness = np.log(self.S / strike)
                time_adj = np.sqrt(time)
                
                # Volatility smile parameters (simplified)
                vol_atm = base_volatility
                vol_skew = -0.1 * moneyness  # Negative skew
                vol_convexity = 0.05 * moneyness**2  # Volatility smile
                
                implied_vol = vol_atm + vol_skew * time_adj + vol_convexity
                implied_vol = max(implied_vol, 0.01)  # Minimum volatility
                
                results.append({
                    'strike': strike,
                    'time_to_expiry': time,
                    'moneyness': moneyness,
                    'implied_volatility': implied_vol
                })
        
        return pd.DataFrame(results)
    
    def risk_sensitivities(self, bump_size=0.01):
        """
        Calculate risk sensitivities using finite difference method.
        
        Args:
            bump_size (float): Size of bump for finite difference calculation
            
        Returns:
            dict: Risk sensitivities
        """
        base_call = self.call_price()
        base_put = self.put_price()
        
        # Price sensitivity (delta verification)
        model_up = BlackScholesModel(self.S * (1 + bump_size), self.K, self.T, self.r, self.sigma, self.q)
        model_down = BlackScholesModel(self.S * (1 - bump_size), self.K, self.T, self.r, self.sigma, self.q)
        
        delta_call_fd = (model_up.call_price() - model_down.call_price()) / (2 * self.S * bump_size)
        delta_put_fd = (model_up.put_price() - model_down.put_price()) / (2 * self.S * bump_size)
        
        # Volatility sensitivity (vega verification)
        model_vol_up = BlackScholesModel(self.S, self.K, self.T, self.r, self.sigma * (1 + bump_size), self.q)
        model_vol_down = BlackScholesModel(self.S, self.K, self.T, self.r, self.sigma * (1 - bump_size), self.q)
        
        vega_call_fd = (model_vol_up.call_price() - model_vol_down.call_price()) / (2 * self.sigma * bump_size)
        vega_put_fd = (model_vol_up.put_price() - model_vol_down.put_price()) / (2 * self.sigma * bump_size)
        
        # Interest rate sensitivity (rho verification)
        model_r_up = BlackScholesModel(self.S, self.K, self.T, self.r * (1 + bump_size), self.sigma, self.q)
        model_r_down = BlackScholesModel(self.S, self.K, self.T, self.r * (1 - bump_size), self.sigma, self.q)
        
        rho_call_fd = (model_r_up.call_price() - model_r_down.call_price()) / (2 * self.r * bump_size)
        rho_put_fd = (model_r_up.put_price() - model_r_down.put_price()) / (2 * self.r * bump_size)
        
        return {
            'call': {
                'delta_fd': delta_call_fd,
                'delta_analytical': self.delta('call'),
                'vega_fd': vega_call_fd,
                'vega_analytical': self.vega(),
                'rho_fd': rho_call_fd,
                'rho_analytical': self.rho('call')
            },
            'put': {
                'delta_fd': delta_put_fd,
                'delta_analytical': self.delta('put'),
                'vega_fd': vega_put_fd,
                'vega_analytical': self.vega(),
                'rho_fd': rho_put_fd,
                'rho_analytical': self.rho('put')
            }
        }