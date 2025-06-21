"""
Market Regime Detection and Analysis

This module implements various techniques for detecting and analyzing
market regimes including:
- Hidden Markov Models (HMM)
- Regime-switching models
- Volatility clustering detection
- Structural break detection
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')


class RegimeDetector:
    """
    Advanced market regime detection using multiple methodologies.
    """
    
    def __init__(self, returns_data, lookback_window=252):
        """
        Initialize regime detector.
        
        Args:
            returns_data (pd.Series or pd.DataFrame): Return series data
            lookback_window (int): Lookback window for rolling calculations
        """
        self.returns = returns_data
        self.lookback = lookback_window
        
        if isinstance(returns_data, pd.DataFrame):
            self.returns = returns_data.iloc[:, 0]  # Use first column if DataFrame
        
        # Calculate features for regime detection
        self._calculate_features()
    
    def _calculate_features(self):
        """Calculate features used for regime detection."""
        # Rolling volatility
        self.volatility = self.returns.rolling(window=21).std() * np.sqrt(252)
        
        # Rolling skewness and kurtosis
        self.skewness = self.returns.rolling(window=63).skew()
        self.kurtosis = self.returns.rolling(window=63).kurt()
        
        # VIX-like volatility measure
        self.vix_proxy = self.returns.rolling(window=21).std() * np.sqrt(252) * 100
        
        # Trend measures
        self.momentum = self.returns.rolling(window=21).mean() * 252
        self.trend_strength = abs(self.returns.rolling(window=63).mean() * 252)
        
        # Market stress indicators
        self.max_drawdown_rolling = self._rolling_max_drawdown()
        self.correlation_breakdown = self._correlation_breakdown()
    
    def _rolling_max_drawdown(self):
        """Calculate rolling maximum drawdown."""
        rolling_max_dd = []
        
        for i in range(len(self.returns)):
            if i < self.lookback:
                rolling_max_dd.append(np.nan)
                continue
            
            window_returns = self.returns.iloc[i-self.lookback:i+1]
            cumulative = (1 + window_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min()
            rolling_max_dd.append(max_dd)
        
        return pd.Series(rolling_max_dd, index=self.returns.index)
    
    def _correlation_breakdown(self):
        """Calculate correlation breakdown indicator."""
        # This is a simplified version - in practice, you'd use multiple assets
        # Here we use autocorrelation as a proxy
        rolling_autocorr = []
        
        for i in range(len(self.returns)):
            if i < 63:
                rolling_autocorr.append(np.nan)
                continue
            
            window_returns = self.returns.iloc[i-63:i+1]
            autocorr = window_returns.autocorr(lag=1)
            rolling_autocorr.append(autocorr if not np.isnan(autocorr) else 0)
        
        return pd.Series(rolling_autocorr, index=self.returns.index)
    
    def gaussian_mixture_regimes(self, n_regimes=3, features=None):
        """
        Detect regimes using Gaussian Mixture Models.
        
        Args:
            n_regimes (int): Number of regimes to detect
            features (list, optional): Features to use for clustering
            
        Returns:
            dict: Regime detection results
        """
        if features is None:
            features = ['volatility', 'momentum', 'skewness']
        
        # Prepare feature matrix
        feature_data = []
        for feature in features:
            if feature == 'volatility':
                feature_data.append(self.volatility)
            elif feature == 'momentum':
                feature_data.append(self.momentum)
            elif feature == 'skewness':
                feature_data.append(self.skewness)
            elif feature == 'kurtosis':
                feature_data.append(self.kurtosis)
            elif feature == 'max_drawdown':
                feature_data.append(self.max_drawdown_rolling)
        
        feature_matrix = pd.concat(feature_data, axis=1)
        feature_matrix.columns = features
        feature_matrix = feature_matrix.dropna()
        
        # Standardize features
        standardized_features = (feature_matrix - feature_matrix.mean()) / feature_matrix.std()
        
        # Fit Gaussian Mixture Model
        gmm = GaussianMixture(n_components=n_regimes, random_state=42)
        regime_labels = gmm.fit_predict(standardized_features)
        regime_probabilities = gmm.predict_proba(standardized_features)
        
        # Create results DataFrame
        results_df = pd.DataFrame(index=feature_matrix.index)
        results_df['regime'] = regime_labels
        
        for i in range(n_regimes):
            results_df[f'prob_regime_{i}'] = regime_probabilities[:, i]
        
        # Calculate regime statistics
        regime_stats = {}
        for regime in range(n_regimes):
            regime_mask = regime_labels == regime
            regime_returns = self.returns.loc[feature_matrix.index][regime_mask]
            
            regime_stats[regime] = {
                'mean_return': regime_returns.mean() * 252,
                'volatility': regime_returns.std() * np.sqrt(252),
                'sharpe_ratio': regime_returns.mean() / regime_returns.std() * np.sqrt(252),
                'skewness': regime_returns.skew(),
                'kurtosis': regime_returns.kurt(),
                'frequency': regime_mask.sum() / len(regime_mask),
                'avg_duration': self._calculate_avg_duration(regime_mask)
            }
        
        return {
            'regimes': results_df,
            'regime_stats': regime_stats,
            'model': gmm,
            'features_used': features,
            'bic': gmm.bic(standardized_features),
            'aic': gmm.aic(standardized_features)
        }
    
    def volatility_clustering_regimes(self):
        """
        Detect regimes based on volatility clustering.
        
        Returns:
            dict: Volatility regime results
        """
        # Use GARCH-like approach to identify high/low volatility regimes
        rolling_vol = self.returns.rolling(window=21).std() * np.sqrt(252)
        
        # Calculate percentiles for regime classification
        vol_25 = rolling_vol.quantile(0.25)
        vol_75 = rolling_vol.quantile(0.75)
        
        # Classify regimes
        regimes = pd.Series(index=rolling_vol.index, dtype=int)
        regimes[rolling_vol <= vol_25] = 0  # Low volatility
        regimes[(rolling_vol > vol_25) & (rolling_vol <= vol_75)] = 1  # Medium volatility
        regimes[rolling_vol > vol_75] = 2  # High volatility
        
        # Calculate persistence (how long each regime lasts)
        regime_persistence = self._calculate_regime_persistence(regimes)
        
        # Calculate regime statistics
        regime_stats = {}
        for regime in [0, 1, 2]:
            regime_names = ['Low Vol', 'Medium Vol', 'High Vol']
            regime_mask = regimes == regime
            regime_returns = self.returns[regime_mask]
            
            if len(regime_returns) > 0:
                regime_stats[regime] = {
                    'name': regime_names[regime],
                    'mean_return': regime_returns.mean() * 252,
                    'volatility': regime_returns.std() * np.sqrt(252),
                    'frequency': regime_mask.sum() / len(regime_mask),
                    'avg_duration': regime_persistence.get(regime, 0),
                    'max_drawdown': self._calculate_regime_max_drawdown(regime_returns)
                }
        
        return {
            'regimes': regimes,
            'regime_stats': regime_stats,
            'volatility_thresholds': {'low': vol_25, 'high': vol_75}
        }
    
    def structural_breaks_detection(self, method='cusum'):
        """
        Detect structural breaks in the time series.
        
        Args:
            method (str): Method to use ('cusum', 'recursive', 'rolling')
            
        Returns:
            dict: Structural break detection results
        """
        if method == 'cusum':
            return self._cusum_test()
        elif method == 'recursive':
            return self._recursive_residuals_test()
        elif method == 'rolling':
            return self._rolling_statistics_test()
        else:
            raise ValueError("Method must be 'cusum', 'recursive', or 'rolling'")
    
    def _cusum_test(self):
        """CUSUM test for structural breaks."""
        # Calculate cumulative sum of standardized returns
        standardized_returns = (self.returns - self.returns.mean()) / self.returns.std()
        cusum = standardized_returns.cumsum()
        
        # Calculate CUSUM statistics
        n = len(cusum)
        cusum_stats = []
        
        for i in range(1, n):
            # Split point statistic
            before = cusum.iloc[:i]
            after = cusum.iloc[i:]
            
            stat = abs(before.iloc[-1] - after.iloc[0])
            cusum_stats.append(stat)
        
        cusum_stats = pd.Series(cusum_stats, index=self.returns.index[1:])
        
        # Find potential break points (peaks in CUSUM statistics)
        peaks, _ = find_peaks(cusum_stats, height=cusum_stats.quantile(0.9))
        break_points = cusum_stats.index[peaks]
        
        return {
            'cusum_statistics': cusum_stats,
            'break_points': break_points,
            'cusum_series': cusum
        }
    
    def _recursive_residuals_test(self):
        """Recursive residuals test for structural stability."""
        # This is a simplified version
        window_size = min(63, len(self.returns) // 4)
        recursive_stats = []
        
        for i in range(window_size, len(self.returns)):
            # Fit model on expanding window
            train_data = self.returns.iloc[:i]
            test_point = self.returns.iloc[i]
            
            # Simple AR(1) model prediction
            if len(train_data) > 1:
                prediction = train_data.iloc[-1]  # Naive prediction
                residual = test_point - prediction
                recursive_stats.append(abs(residual))
            else:
                recursive_stats.append(0)
        
        recursive_stats = pd.Series(recursive_stats, index=self.returns.index[window_size:])
        
        # Identify outliers as potential break points
        threshold = recursive_stats.quantile(0.95)
        break_points = recursive_stats[recursive_stats > threshold].index
        
        return {
            'recursive_residuals': recursive_stats,
            'break_points': break_points,
            'threshold': threshold
        }
    
    def _rolling_statistics_test(self):
        """Rolling statistics test for regime changes."""
        window = 63
        
        # Rolling mean and volatility
        rolling_mean = self.returns.rolling(window=window).mean()
        rolling_vol = self.returns.rolling(window=window).std()
        
        # Calculate changes in statistics
        mean_changes = abs(rolling_mean.diff())
        vol_changes = abs(rolling_vol.diff())
        
        # Normalize changes
        mean_changes_norm = mean_changes / rolling_mean.std()
        vol_changes_norm = vol_changes / rolling_vol.std()
        
        # Combined change indicator
        combined_changes = mean_changes_norm + vol_changes_norm
        
        # Identify significant changes
        threshold = combined_changes.quantile(0.9)
        break_points = combined_changes[combined_changes > threshold].index
        
        return {
            'mean_changes': mean_changes_norm,
            'volatility_changes': vol_changes_norm,
            'combined_changes': combined_changes,
            'break_points': break_points,
            'threshold': threshold
        }
    
    def _calculate_avg_duration(self, regime_mask):
        """Calculate average duration of regime periods."""
        if regime_mask.sum() == 0:
            return 0
        
        durations = []
        current_duration = 0
        
        for is_regime in regime_mask:
            if is_regime:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0
        
        # Don't forget the last duration if it ends with the regime
        if current_duration > 0:
            durations.append(current_duration)
        
        return np.mean(durations) if durations else 0
    
    def _calculate_regime_persistence(self, regimes):
        """Calculate persistence of each regime."""
        persistence = {}
        
        for regime in regimes.unique():
            if pd.isna(regime):
                continue
            
            regime_mask = regimes == regime
            persistence[regime] = self._calculate_avg_duration(regime_mask)
        
        return persistence
    
    def _calculate_regime_max_drawdown(self, regime_returns):
        """Calculate maximum drawdown for a specific regime."""
        if len(regime_returns) < 2:
            return 0
        
        cumulative = (1 + regime_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def regime_switching_model(self, n_regimes=2):
        """
        Fit a simple regime-switching model.
        
        Args:
            n_regimes (int): Number of regimes
            
        Returns:
            dict: Model results
        """
        # This is a simplified Markov regime switching model
        # In practice, you'd use specialized libraries like statsmodels
        
        # Use K-means clustering on returns and volatility
        features = np.column_stack([
            self.returns.values,
            self.volatility.fillna(self.volatility.mean()).values
        ])
        
        # Remove NaN values
        valid_idx = ~np.isnan(features).any(axis=1)
        features_clean = features[valid_idx]
        
        # Fit K-means
        kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        regime_labels = kmeans.fit_predict(features_clean)
        
        # Calculate transition matrix
        transition_matrix = self._calculate_transition_matrix(regime_labels, n_regimes)
        
        # Calculate regime statistics
        regime_stats = {}
        for regime in range(n_regimes):
            regime_mask = regime_labels == regime
            regime_returns = self.returns.iloc[valid_idx][regime_mask]
            
            regime_stats[regime] = {
                'mean_return': regime_returns.mean() * 252,
                'volatility': regime_returns.std() * np.sqrt(252),
                'frequency': regime_mask.sum() / len(regime_mask)
            }
        
        # Create full regime series
        full_regimes = pd.Series(index=self.returns.index, dtype=float)
        full_regimes.iloc[valid_idx] = regime_labels
        
        return {
            'regimes': full_regimes,
            'transition_matrix': transition_matrix,
            'regime_stats': regime_stats,
            'model': kmeans
        }
    
    def _calculate_transition_matrix(self, regime_labels, n_regimes):
        """Calculate regime transition probability matrix."""
        transition_counts = np.zeros((n_regimes, n_regimes))
        
        for i in range(len(regime_labels) - 1):
            current_regime = regime_labels[i]
            next_regime = regime_labels[i + 1]
            transition_counts[current_regime, next_regime] += 1
        
        # Convert to probabilities
        transition_matrix = np.zeros((n_regimes, n_regimes))
        for i in range(n_regimes):
            row_sum = transition_counts[i].sum()
            if row_sum > 0:
                transition_matrix[i] = transition_counts[i] / row_sum
        
        return transition_matrix
    
    def generate_regime_report(self):
        """
        Generate comprehensive regime analysis report.
        
        Returns:
            dict: Complete regime analysis
        """
        report = {}
        
        # Gaussian mixture regimes
        try:
            gmm_results = self.gaussian_mixture_regimes(n_regimes=3)
            report['gaussian_mixture'] = gmm_results
        except Exception as e:
            report['gaussian_mixture'] = {'error': str(e)}
        
        # Volatility clustering regimes
        try:
            vol_results = self.volatility_clustering_regimes()
            report['volatility_clustering'] = vol_results
        except Exception as e:
            report['volatility_clustering'] = {'error': str(e)}
        
        # Structural breaks
        try:
            break_results = self.structural_breaks_detection('cusum')
            report['structural_breaks'] = break_results
        except Exception as e:
            report['structural_breaks'] = {'error': str(e)}
        
        # Regime switching model
        try:
            switching_results = self.regime_switching_model()
            report['regime_switching'] = switching_results
        except Exception as e:
            report['regime_switching'] = {'error': str(e)}
        
        return report