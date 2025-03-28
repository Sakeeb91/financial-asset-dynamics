# Heston Stochastic Volatility Model Simulation Results

## Model Overview

The Heston model is a stochastic volatility model used for modeling asset price dynamics. Unlike simpler models like Geometric Brownian Motion (GBM), the Heston model accounts for stochastic (time-varying) volatility, which better captures real-world market behavior such as volatility clustering and skewed returns.

The model consists of two coupled stochastic differential equations (SDEs):

1. Asset price process: $dS_t = \mu S_t dt + \sqrt{v_t} S_t dW_1$
2. Variance process: $dv_t = \kappa(\theta - v_t) dt + \xi \sqrt{v_t} dW_2$

Where:
- $S_t$ is the asset price at time $t$
- $v_t$ is the variance (volatility squared) at time $t$
- $\mu$ is the drift rate (expected return)
- $\kappa$ is the rate of mean reversion for variance
- $\theta$ is the long-term mean of variance
- $\xi$ is the volatility of volatility (how volatile is the volatility itself)
- $\rho$ is the correlation between the two Wiener processes $dW_1$ and $dW_2$

## Simulation Parameters

For this simulation, we used the following parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\mu$ | 0.10 | Drift coefficient (10% annual expected return) |
| $\kappa$ | 2.0 | Mean reversion speed for variance |
| $\theta$ | 0.04 | Long-term variance (corresponds to 20% volatility) |
| $\xi$ | 0.3 | Volatility of volatility |
| $\rho$ | -0.7 | Correlation between price and volatility (-0.7 is typical) |
| $S_0$ | $100 | Initial asset price |
| $v_0$ | 0.04 | Initial variance (corresponds to 20% volatility) |

## Simulation Results

### 1. Single Path Analysis

![Single Path](../plots/heston_single_path.png)

This plot shows a single realization of the Heston model over a one-year period. The top panel shows the asset price, while the bottom panel shows the corresponding volatility path. 

Key observations:
- The price path shows the stochastic nature of asset prices
- The volatility fluctuates around its long-term mean (20%)
- The final price was $111.57 (11.57% return over one year)
- The final volatility was 15.84%, which is below the long-term volatility of 20%

### 2. Multiple Paths Analysis

![Multiple Paths](../plots/heston_multiple_paths.png)

This visualization shows 50 different simulated paths of the asset price and volatility. The red line in the price chart represents the mean price path, while the blue line in the volatility chart represents the mean volatility.

Key observations:
- There is significant dispersion in possible price outcomes
- Volatility reverts to its long-term mean (20%)
- The mean price path shows the expected upward trend due to the positive drift

### 3. Distribution of Final Prices

![Price Distribution](../plots/heston_distribution.png)

This histogram shows the distribution of final prices after one year, based on 10,000 simulated paths.

Key statistics:
- Mean: $110.34 (10.34% average return)
- Standard Deviation: $21.01
- 5th Percentile: $73.91 (worst-case scenarios)
- 95th Percentile: $143.08 (best-case scenarios)
- Skewness: -0.2118 (slightly negatively skewed)
- Excess Kurtosis: 0.1154 (slightly fatter tails than normal distribution)

The distribution is compared with a normal distribution with the same mean and standard deviation. The actual distribution has slightly fatter tails and is slightly negatively skewed, which is consistent with real market returns.

### 4. Parameter Comparison

![Parameter Comparison](../plots/heston_parameter_comparison.png)

This chart compares how different volatility and correlation parameters affect the price and volatility paths. Four alternative parameter sets were tested:

1. **Low Vol of Vol** ($\xi = 0.1$): Less volatile volatility, resulting in smoother volatility paths
2. **High Vol of Vol** ($\xi = 0.5$): More volatile volatility, creating more extreme swings
3. **No Correlation** ($\rho = 0$): No relationship between price and volatility movements
4. **Positive Correlation** ($\rho = 0.5$): Prices and volatility tend to move in the same direction

Key observations:
- Higher volatility of volatility ($\xi$) leads to more extreme price paths
- Positive correlation between price and volatility ($\rho > 0$) tends to amplify upward price movements
- The base model (negative correlation) tends to dampen extreme price movements

### 5. Volatility Clustering

![Volatility Clustering](../plots/heston_volatility_clustering.png)

This plot demonstrates volatility clustering over a 5-year period. Volatility clustering is the tendency of high volatility periods to be followed by more high volatility, and low volatility periods to be followed by more low volatility.

The simulation uses a higher volatility of volatility parameter ($\xi = 0.5$) to make the clustering more pronounced.

Key observations:
- Clear periods of high and low volatility
- Daily log returns show larger fluctuations during high volatility periods
- The volatility process is mean-reverting, always eventually returning to the long-term level

### 6. Option Pricing

![Option Pricing](../plots/heston_option_pricing.png)

This analysis compares option prices under the Heston model with those calculated using the Black-Scholes model. Options with different strike prices were priced for a 1-year maturity.

Key observations:
- The Heston model generates a volatility smile/skew, with implied volatility higher for out-of-the-money options
- For at-the-money options (strike = $100), both models produce similar prices
- For out-of-the-money options, especially puts, the Heston model predicts higher prices than Black-Scholes
- This reflects the model's ability to capture tail risk better than Black-Scholes

## Conclusions

The Heston stochastic volatility model provides several advantages over simpler models:

1. **Volatility Clustering**: The model captures the tendency of volatility to cluster, which is observed in real markets.

2. **Fat Tails and Skewness**: The distribution of returns shows slightly fat tails and negative skewness, consistent with empirical observations.

3. **Volatility Smile**: The model reproduces the volatility smile/skew observed in option markets, where implied volatility varies with strike price.

4. **Leverage Effect**: The negative correlation between price and volatility ($\rho = -0.7$) captures the leverage effect, where falling prices tend to be associated with rising volatility.

These features make the Heston model valuable for risk management, option pricing, and understanding market dynamics beyond what simpler models like Geometric Brownian Motion can provide.

## Limitations

Despite its advantages, the Heston model has some limitations:

1. It requires estimation of more parameters than simpler models, which can be challenging.
2. The model assumes that volatility follows a mean-reverting process, which may not always be true in all market conditions.
3. In extreme market conditions, even the Heston model may underestimate tail risks.

Future work could explore model extensions that incorporate jumps in both price and volatility processes for even more realistic simulations. 