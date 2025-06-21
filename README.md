# Financial Asset Dynamics - Advanced Quantitative Finance Library

A comprehensive Python library for advanced financial modeling, portfolio optimization, risk management, and derivatives pricing. This project demonstrates sophisticated quantitative finance techniques used by institutional investors and hedge funds.

## 🎯 Overview

This project implements cutting-edge financial mathematics and quantitative analysis techniques including:

- **Stochastic Differential Equations (SDEs)**: Multiple models for asset price dynamics
- **Portfolio Optimization**: Modern portfolio theory with advanced constraints
- **Risk Management**: Comprehensive VaR, CVaR, and stress testing
- **Options Pricing**: Black-Scholes with Greeks and volatility surfaces
- **Market Regime Detection**: Advanced statistical techniques for market analysis
- **Real-Time Market Data**: Integration with financial APIs

## 🚀 Advanced Features

### 📊 **Stochastic Models**
- **Geometric Brownian Motion (GBM)**: Classic asset price modeling
- **Heston Model**: Stochastic volatility with correlation effects
- **SABR Model**: Volatility smile modeling for derivatives
- **Jump-Diffusion**: Market shock and crash modeling
- **Ornstein-Uhlenbeck**: Mean-reverting processes for rates and commodities

### 🎯 **Portfolio Optimization**
- **Markowitz Optimization**: Mean-variance framework
- **Black-Litterman Model**: Bayesian approach with investor views
- **Risk Parity**: Equal risk contribution portfolios
- **Maximum Sharpe Ratio**: Risk-adjusted return optimization
- **CVaR Optimization**: Downside risk-focused strategies

### ⚠️ **Risk Management**
- **Value at Risk (VaR)**: Historical, parametric, and Monte Carlo methods
- **Conditional VaR (Expected Shortfall)**: Tail risk quantification
- **Maximum Drawdown Analysis**: Worst-case scenario planning
- **Stress Testing**: Custom scenario analysis
- **GARCH Volatility Modeling**: Time-varying volatility

### 📈 **Options Analytics**
- **Black-Scholes Pricing**: European and American options
- **Greeks Calculation**: Delta, Gamma, Theta, Vega, Rho
- **Implied Volatility**: Market-implied volatility extraction
- **Volatility Surfaces**: 3D volatility smile modeling
- **Option Chains**: Complete strike/expiry analysis

### 🔄 **Market Intelligence**
- **Regime Detection**: Hidden Markov Models for market states
- **Structural Break Analysis**: Statistical change point detection
- **Correlation Dynamics**: Time-varying correlation analysis
- **Market Stress Indicators**: Systemic risk measurement

## 💻 Installation

```bash
# Clone the repository
git clone https://github.com/Sakeeb91/financial-asset-dynamics.git
cd financial-asset-dynamics

# Install dependencies
pip install -r sde_asset_modeling/requirements.txt

# Install the package in development mode
cd sde_asset_modeling
pip install -e .
```

## 🚀 Quick Start Guide

### 1. Portfolio Optimization

```python
from sde_asset_modeling.portfolio.optimizer import PortfolioOptimizer
import pandas as pd

# Load your returns data
returns_data = pd.read_csv('your_returns.csv', index_col=0, parse_dates=True)

# Initialize optimizer
optimizer = PortfolioOptimizer(returns_data)

# Maximum Sharpe ratio portfolio
max_sharpe = optimizer.maximum_sharpe_optimization()
print(f"Optimal weights: {max_sharpe['weights']}")
print(f"Expected return: {max_sharpe['expected_return']:.2%}")
print(f"Sharpe ratio: {max_sharpe['sharpe_ratio']:.3f}")

# Generate efficient frontier
frontier = optimizer.efficient_frontier(n_points=100)
```

### 2. Risk Analysis

```python
from sde_asset_modeling.portfolio.risk_metrics import RiskMetrics

# Initialize risk calculator
risk_calc = RiskMetrics(returns_data)

# Calculate comprehensive risk metrics
risk_report = risk_calc.generate_risk_report()

# Value at Risk
var_95 = risk_calc.value_at_risk(confidence_level=0.05)
print(f"VaR (95%): {var_95}")

# Stress testing
stress_scenarios = {
    'market_crash': {'factor': -0.3},
    'volatility_spike': {'vol_mult': 2}
}
stress_results = risk_calc.stress_testing(stress_scenarios)
```

### 3. Options Pricing

```python
from sde_asset_modeling.options.black_scholes import BlackScholesModel

# Create Black-Scholes model
bs = BlackScholesModel(
    spot_price=100,
    strike_price=105,
    time_to_expiry=0.25,  # 3 months
    risk_free_rate=0.05,
    volatility=0.2
)

# Calculate option prices and Greeks
call_price = bs.call_price()
put_price = bs.put_price()
greeks = bs.all_greeks('call')

print(f"Call price: ${call_price:.2f}")
print(f"Delta: {greeks['delta']:.3f}")
print(f"Gamma: {greeks['gamma']:.3f}")

# Generate option chain
strikes = range(90, 111)
option_chain = bs.option_chain(strikes)
```

### 4. Market Regime Detection

```python
from sde_asset_modeling.analytics.regime_detection import RegimeDetector

# Initialize regime detector
detector = RegimeDetector(returns_data.iloc[:, 0])  # Single asset

# Detect volatility regimes
vol_regimes = detector.volatility_clustering_regimes()
print("Regime statistics:", vol_regimes['regime_stats'])

# Gaussian mixture model regimes
gmm_regimes = detector.gaussian_mixture_regimes(n_regimes=3)
```

### 5. Advanced Dashboards

```bash
# Create comprehensive financial dashboard
python sde_asset_modeling/create_advanced_dashboard.py

# Generate professional HTML report
python sde_asset_modeling/create_comprehensive_report.py
```

## 📁 Project Architecture

```
financial-asset-dynamics/
├── sde_asset_modeling/
│   ├── src/sde_asset_modeling/
│   │   ├── models/              # SDE implementations
│   │   │   ├── gbm.py          # Geometric Brownian Motion
│   │   │   ├── heston.py       # Heston stochastic volatility
│   │   │   ├── sabr.py         # SABR model
│   │   │   └── jump_diffusion.py
│   │   ├── portfolio/          # Portfolio optimization
│   │   │   ├── optimizer.py    # Advanced optimization strategies
│   │   │   └── risk_metrics.py # Comprehensive risk analysis
│   │   ├── options/            # Options pricing
│   │   │   ├── black_scholes.py # Complete B-S implementation
│   │   │   └── greeks.py       # Greeks calculations
│   │   ├── analytics/          # Market intelligence
│   │   │   ├── regime_detection.py # Market regime analysis
│   │   │   ├── market_data.py  # Real-time data integration
│   │   │   └── backtesting.py  # Strategy backtesting
│   │   ├── simulation/         # Monte Carlo engines
│   │   └── utils/              # Plotting and utilities
│   ├── create_advanced_dashboard.py    # Interactive dashboards
│   ├── create_comprehensive_report.py  # Professional reports
│   └── requirements.txt        # Dependencies
└── README.md
```

## 🎯 Key Differentiators

### **Professional-Grade Implementation**
- **Institutional Quality**: Code follows best practices used in quantitative hedge funds
- **Performance Optimized**: Efficient algorithms suitable for large-scale analysis
- **Comprehensive Testing**: Robust implementation with error handling
- **Real Market Data**: Integration with live financial data feeds

### **Advanced Mathematical Models**
- **Multi-Factor Models**: Beyond basic single-factor approaches
- **Regime-Aware Analysis**: Market state-dependent modeling
- **Non-Gaussian Distributions**: Fat-tail and skewness modeling
- **Correlation Dynamics**: Time-varying correlation structures

### **Enterprise Features**
- **Risk Management**: Bank-grade VaR and stress testing
- **Portfolio Construction**: Institutional-level optimization
- **Regulatory Compliance**: Risk metrics aligned with Basel III
- **Scalable Architecture**: Designed for production environments

## 📊 Demonstration Outputs

The project generates professional-quality outputs including:

- **Interactive HTML Reports**: Comprehensive financial analysis
- **Publication-Ready Charts**: High-quality visualizations
- **Risk Dashboards**: Real-time risk monitoring
- **Performance Attribution**: Detailed return analysis
- **Stress Test Results**: Scenario analysis outcomes

## 🎓 Educational Value

This project demonstrates mastery of:

- **Quantitative Finance Theory**: Advanced mathematical models
- **Software Engineering**: Clean, maintainable code architecture
- **Data Science**: Statistical analysis and machine learning
- **Risk Management**: Comprehensive risk framework
- **Financial Markets**: Deep understanding of market dynamics

## 🔧 Dependencies

```python
# Core scientific computing
numpy>=1.20.0
scipy>=1.7.0
pandas>=1.3.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0

# Financial data
yfinance>=0.1.70

# Optimization
cvxpy>=1.2.0

# Machine learning
scikit-learn>=1.0.0

# Time series analysis
arch>=5.0.0
statsmodels>=0.13.0

# Optional: Production features
quantlib>=1.25
```

## 🚀 Getting Started

1. **Clone and Install**
   ```bash
   git clone https://github.com/Sakeeb91/financial-asset-dynamics.git
   cd financial-asset-dynamics/sde_asset_modeling
   pip install -r requirements.txt
   ```

2. **Run Advanced Analytics**
   ```bash
   python create_advanced_dashboard.py
   python create_comprehensive_report.py
   ```

3. **Explore Examples**
   - Check generated HTML reports
   - Review Jupyter notebooks
   - Examine visualization outputs

## 📈 Performance Highlights

- **Multi-Asset Analysis**: Handle hundreds of securities simultaneously
- **Real-Time Processing**: Live market data integration
- **Advanced Optimization**: Solve complex portfolio problems
- **Comprehensive Risk**: Bank-grade risk measurement
- **Professional Reporting**: Institutional-quality outputs

## 🎯 Target Applications

- **Portfolio Management**: Institutional investment strategies
- **Risk Management**: Comprehensive risk framework
- **Derivatives Trading**: Options pricing and Greeks
- **Research & Development**: Advanced financial modeling
- **Academic Projects**: Quantitative finance education

## 📚 License & Contributing

**License**: MIT License - feel free to use for educational and commercial purposes

**Contributing**: Contributions welcome! This project showcases advanced quantitative finance techniques and welcomes improvements from the community.

## 🏆 Acknowledgements

This project implements state-of-the-art techniques from quantitative finance literature and demonstrates institutional-quality financial modeling capabilities. Perfect for showcasing advanced quantitative skills to potential employers in finance, hedge funds, and fintech companies. 