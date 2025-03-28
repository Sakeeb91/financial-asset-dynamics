# SABR Model Simulation Results

## Overview

The SABR (Stochastic Alpha, Beta, Rho) model is a stochastic volatility model for the evolution of forward rates, particularly useful for modeling interest rate derivatives. It was introduced by Hagan et al. in 2002 and has become a standard tool in the financial industry due to its ability to capture the volatility smile/skew observed in market data.

The SABR model is defined by the following stochastic differential equations (SDEs):

- Forward rate process: $dF_t = \alpha_t F_t^\beta dW_t^1$
- Volatility process: $d\alpha_t = \nu \alpha_t dW_t^2$
- Where $dW_t^1$ and $dW_t^2$ are Wiener processes with correlation $\rho$

## Model Parameters

The SABR model has several key parameters that control its behavior:

- **Initial Forward Rate (F₀)**: 3.00% - The starting value of the forward rate.
- **Initial Volatility (α₀)**: 0.15 - The starting value of the volatility.
- **CEV Parameter (β)**: 0.50 - Controls the relationship between volatility and the level of the forward rate.
- **Volatility of Volatility (ν)**: 0.40 - Controls how much the volatility itself varies.
- **Correlation (ρ)**: -0.30 - Correlation between the forward rate and volatility processes.

## Simulation Results

### Single Path Simulation

The first simulation shows a single realization of the SABR model over a one-year period:

![SABR Single Path](../../plots/sabr/sabr_single_path.png)

The simulation demonstrates how both the forward rate and its volatility evolve stochastically over time. Note that in the risk-neutral measure, the forward rate has no drift term, while the volatility follows a log-normal process.

### Multiple Paths Simulation

To better understand the distribution of possible outcomes, we simulated 20 different paths of the SABR model:

![SABR Multiple Paths](../../plots/sabr/sabr_multiple_paths.png)

The simulation shows the range of possible paths for both the forward rate and volatility. The mean path is highlighted, showing the expected evolution of the model. Note the "fanning out" of paths over time, demonstrating increasing uncertainty.

### Volatility Smile/Skew Analysis

One of the key features of the SABR model is its ability to capture the volatility smile/skew observed in interest rate option markets:

![SABR Volatility Smile](../../plots/sabr/sabr_volatility_smile.png)

The plot shows implied volatilities for different strike rates and option maturities. The SABR model naturally produces a smile pattern (higher implied volatilities for in-the-money and out-of-the-money options compared to at-the-money options), which matches market observations.

### Parameter Analysis

#### Beta Parameter (β)

The beta parameter controls the relationship between volatility and the level of the forward rate:

![SABR Beta Comparison](../../plots/sabr/sabr_beta_comparison.png)

- **β = 0**: Normal model (volatility independent of rate level)
- **β = 0.5**: CEV model (commonly used for interest rates)
- **β = 1**: Log-normal model (volatility proportional to rate level)

The parameter significantly affects the shape of the volatility smile, with higher β values typically producing more pronounced skew effects.

#### Volatility of Volatility Parameter (ν)

The nu parameter controls how much the volatility itself varies:

![SABR Nu Comparison](../../plots/sabr/sabr_nu_comparison.png)

Higher values of ν lead to more pronounced volatility smiles, as the increased uncertainty in volatility affects the tails of the distribution more significantly than the center.

#### Correlation Parameter (ρ)

The correlation between forward rate and volatility processes has a significant impact on the skew:

![SABR Rho Comparison](../../plots/sabr/sabr_rho_comparison.png)

Negative correlation (commonly observed in interest rate markets) produces a downward sloping skew, while positive correlation creates an upward sloping skew.

### Option Pricing

The SABR model provides more accurate option prices compared to simpler models like Black's model with constant volatility:

![SABR Option Prices](../../plots/sabr/sabr_option_prices.png)

The plots compare prices for caps (call options on forward rates) and floors (put options on forward rates) between the SABR model and Black's model with constant volatility. The SABR model captures the volatility smile effect, resulting in higher prices for out-of-the-money options compared to Black's model.

## Dashboard Summary

The SABR dashboard combines all simulation results in a single view:

![SABR Dashboard](../../plots/sabr/sabr_dashboard.png)

This dashboard provides a comprehensive visualization of the SABR model's behavior and its implications for option pricing and risk management.

## Conclusion

The SABR model offers several advantages for modeling interest rate derivatives:

1. **Volatility Smile Capture**: Accurately reproduces the market-observed volatility smile/skew.
2. **Flexible Parameters**: Can be calibrated to match various market conditions.
3. **Analytical Tractability**: Provides closed-form approximations for implied volatilities.
4. **Mean Reversion**: Can capture mean reverting behavior observed in interest rates.

These features make the SABR model a valuable tool for pricing and risk management of interest rate derivatives such as caps, floors, and swaptions. 