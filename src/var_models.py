"""
Value-at-Risk (VaR) and Expected Shortfall (ES) Calculation Models
===================================================================

This module implements core market risk metrics used in financial risk management:

Risk Metrics Implemented:
-------------------------
1. Parametric VaR (Variance-Covariance Method)
2. Historical Simulation VaR
3. Monte Carlo VaR
4. Expected Shortfall (ES/CVaR) - all three methods
5. VaR Scaling (square-root-of-time rule)
6. Marginal VaR and Component VaR (risk decomposition)

Regulatory Context:
-------------------
- Basel III: Requires 99% VaR with 10-day holding period
- FRTB/Basel IV: Requires 97.5% Expected Shortfall as primary metric
- Backtesting: 250-day rolling window for model validation

Key Conventions:
----------------
- VaR is returned as a NEGATIVE number (representing potential loss)
- Confidence levels: 0.99 = 99% confidence (1% tail)
- Returns are assumed to be log returns (continuously compounded)

Mathematical References:
------------------------
- Jorion, P. "Value at Risk: The New Benchmark for Managing Financial Risk"
- Hull, J.C. "Risk Management and Financial Institutions"
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


# =============================================================================
# PARAMETRIC (VARIANCE-COVARIANCE) VAR
# =============================================================================

def parametric_var(returns, confidence=0.99):
    """
    Calculate Parametric (Variance-Covariance) VaR.

    This method assumes returns follow a normal distribution and calculates
    VaR using the mean (mu) and standard deviation (sigma) of returns.

    Mathematical Formula:
    ---------------------
    VaR = μ + z_α * σ

    where:
        μ = mean of returns
        σ = standard deviation of returns
        z_α = z-score at (1-confidence) quantile
              For 99% confidence: z_0.01 = -2.326
              For 95% confidence: z_0.05 = -1.645

    Pros:
    -----
    - Computationally efficient (closed-form solution)
    - Easy to implement and explain
    - Works well for liquid, normally distributed assets

    Cons:
    -----
    - Assumes normality (underestimates tail risk for fat-tailed distributions)
    - Does not capture skewness or excess kurtosis
    - May underestimate VaR during market stress

    Parameters:
    -----------
    returns : pd.Series
        Historical returns (log returns preferred)
    confidence : float
        Confidence level (e.g., 0.99 for 99%)

    Returns:
    --------
    float
        VaR as a negative number (potential loss)
        Example: -0.0441 means 4.41% potential loss
    """
    # Calculate mean (expected return) from historical data
    mu = returns.mean()

    # Calculate standard deviation (volatility) from historical data
    sigma = returns.std()

    # Get z-score for the left tail at (1-confidence) probability
    # norm.ppf(0.01) = -2.326 for 99% confidence
    # This is the inverse CDF (quantile function) of standard normal
    z_score = norm.ppf(1 - confidence)

    # VaR = mu + z_score * sigma
    # Since z_score is negative, this gives us the left tail threshold
    return mu + z_score * sigma


# =============================================================================
# HISTORICAL SIMULATION VAR
# =============================================================================

def historical_var(returns, confidence=0.99):
    """
    Calculate Historical Simulation VaR.

    This non-parametric method uses the empirical distribution of historical
    returns without making any distributional assumptions. VaR is simply
    the percentile of the historical return distribution.

    Mathematical Formula:
    ---------------------
    VaR = Percentile(returns, (1-α) × 100)

    where:
        α = confidence level (e.g., 0.99)
        For 99% VaR: take the 1st percentile

    Pros:
    -----
    - No distributional assumptions (captures fat tails naturally)
    - Captures actual historical tail behavior
    - Simple and intuitive interpretation
    - Better for non-normal return distributions

    Cons:
    -----
    - Depends heavily on historical sample (ghost effects)
    - Cannot predict losses beyond historical worst case
    - Sensitive to sample period selection
    - "History may not repeat itself"

    Parameters:
    -----------
    returns : pd.Series or array
        Historical returns
    confidence : float
        Confidence level (e.g., 0.99 for 99%)

    Returns:
    --------
    float
        VaR as the percentile of historical returns
    """
    # Calculate the (1-confidence)*100 percentile
    # For 99% confidence: percentile at 1% (left tail)
    # np.percentile(data, 1) gives the value below which 1% of data falls
    return np.percentile(returns, (1 - confidence) * 100)


# =============================================================================
# MONTE CARLO SIMULATION VAR
# =============================================================================

def monte_carlo_var(returns, confidence=0.99, simulations=10000, return_simulated=False):
    """
    Calculate Monte Carlo Simulation VaR.

    This method generates a large number of random return scenarios based on
    the estimated distribution parameters, then calculates VaR from the
    simulated distribution. Here we use normal distribution, but this method
    can be extended to use other distributions (t-distribution, etc.).

    Mathematical Process:
    ---------------------
    1. Estimate μ and σ from historical returns
    2. Generate N random draws: r_sim ~ N(μ, σ)
    3. VaR = Percentile(r_sim, (1-α) × 100)

    Pros:
    -----
    - Flexible: can use any distribution (normal, t, etc.)
    - Can incorporate complex dependencies and correlations
    - Provides full distribution of potential outcomes
    - Can model path-dependent instruments (options, etc.)

    Cons:
    -----
    - Computationally intensive
    - Results depend on assumed distribution
    - Random seed affects reproducibility
    - Model risk if wrong distribution assumed

    Parameters:
    -----------
    returns : pd.Series
        Historical returns used to estimate parameters
    confidence : float
        Confidence level
    simulations : int
        Number of Monte Carlo simulations (default 10,000)
        Higher = more stable results but slower
    return_simulated : bool
        Whether to return the simulated returns array

    Returns:
    --------
    float or tuple
        VaR (and optionally simulated returns array)
    """
    # Estimate distribution parameters from historical data
    mu = returns.mean()
    sigma = returns.std()

    # Generate N random draws from normal distribution
    # np.random.normal(mean, std, size) generates random samples
    simulated = np.random.normal(mu, sigma, simulations)

    # Calculate VaR as percentile of simulated distribution
    var = np.percentile(simulated, (1 - confidence) * 100)

    if return_simulated:
        return var, simulated
    return var


# =============================================================================
# EXPECTED SHORTFALL (ES) / CONDITIONAL VAR (CVaR)
# =============================================================================

def parametric_es(returns, confidence=0.975):
    """
    Calculate Parametric Expected Shortfall (ES/CVaR).

    Expected Shortfall (also called Conditional VaR or CVaR) measures the
    AVERAGE loss given that the loss exceeds VaR. This is the required risk
    metric under FRTB (Basel IV) at 97.5% confidence level.

    Mathematical Formula (under normality):
    ---------------------------------------
    ES = μ - σ × φ(z_α) / (1 - α)

    where:
        μ = mean of returns
        σ = standard deviation of returns
        φ(z) = PDF of standard normal (probability density function)
        z_α = z-score at (1-α) quantile
        α = confidence level

    Why ES is preferred over VaR (FRTB rationale):
    ----------------------------------------------
    1. ES is a "coherent" risk measure (satisfies subadditivity)
    2. ES captures tail severity, not just threshold
    3. ES does not incentivize concentration in tail risks

    FRTB Requirement:
    -----------------
    97.5% ES is roughly equivalent to 99% VaR for normal distributions,
    but better captures tail risk for fat-tailed distributions.

    Parameters:
    -----------
    returns : pd.Series
        Historical returns
    confidence : float
        Confidence level (e.g., 0.975 for 97.5% - FRTB standard)

    Returns:
    --------
    float
        ES as a negative number (expected tail loss)
    """
    mu = returns.mean()
    sigma = returns.std()

    # Get z-score for the confidence level
    z_alpha = norm.ppf(1 - confidence)

    # Get PDF (probability density) value at the z-score
    # This is the height of the normal curve at the VaR threshold
    phi_z = norm.pdf(z_alpha)

    # ES formula under normality assumption
    # The term phi_z / (1 - confidence) accounts for the average
    # of the conditional distribution beyond the threshold
    return mu - sigma * phi_z / (1 - confidence)


def historical_es(returns, confidence=0.975):
    """
    Calculate Historical Expected Shortfall.

    This method calculates ES as the simple average of all returns that
    fall below the VaR threshold. No distributional assumptions required.

    Mathematical Formula:
    ---------------------
    ES = E[r | r < VaR] = (1/n_tail) × Σ r_i, for all r_i < VaR

    This is the most intuitive interpretation: "What is the average loss
    when we have a bad day (worse than VaR)?"

    Relationship to VaR:
    --------------------
    ES is ALWAYS >= VaR (in absolute terms) because ES averages all
    losses beyond VaR, while VaR is just the threshold.

    Parameters:
    -----------
    returns : pd.Series or array
        Historical returns
    confidence : float
        Confidence level

    Returns:
    --------
    float
        ES as average of tail losses (always more negative than VaR)
    """
    # First, calculate the VaR threshold
    var = historical_var(returns, confidence)

    # Select all returns that are worse than (below) VaR
    if isinstance(returns, pd.Series):
        tail_returns = returns[returns <= var]
    else:
        tail_returns = returns[returns <= var]

    # ES = average of all tail losses
    return np.mean(tail_returns)


def monte_carlo_es(returns, confidence=0.975, simulations=10000):
    """
    Calculate Monte Carlo Expected Shortfall.

    Combines Monte Carlo simulation with ES calculation. First generates
    simulated returns, then calculates ES from the simulated distribution.

    Process:
    --------
    1. Estimate μ, σ from historical returns
    2. Generate N simulated returns from N(μ, σ)
    3. Calculate VaR from simulated distribution
    4. ES = average of simulated returns below VaR

    Parameters:
    -----------
    returns : pd.Series
        Historical returns
    confidence : float
        Confidence level
    simulations : int
        Number of simulations

    Returns:
    --------
    tuple
        (ES, VaR, simulated_returns)
        Returns all three for analysis and visualization
    """
    mu = returns.mean()
    sigma = returns.std()

    # Generate simulated returns
    simulated = np.random.normal(mu, sigma, simulations)

    # Calculate VaR from simulated distribution
    var = np.percentile(simulated, (1 - confidence) * 100)

    # Select tail (losses beyond VaR)
    tail = simulated[simulated <= var]

    # ES = average of tail losses
    es = tail.mean()

    return es, var, simulated


# =============================================================================
# VAR SCALING (SQUARE-ROOT-OF-TIME RULE)
# =============================================================================

def scale_var(var_1d, holding_period):
    """
    Scale 1-day VaR to different holding periods using square-root-of-time rule.

    Mathematical Formula:
    ---------------------
    VaR_T = VaR_1d × √T

    where T = holding period in days

    Assumptions (important!):
    -------------------------
    This scaling rule assumes:
    1. Returns are IID (independent and identically distributed)
    2. No mean reversion in returns
    3. No serial correlation

    These assumptions may not hold in practice, especially during crises
    when volatility clusters and correlations increase.

    Regulatory Context:
    -------------------
    - Basel III: Requires 10-day VaR for market risk capital
    - VaR_10d = VaR_1d × √10 ≈ VaR_1d × 3.162

    When this rule may fail:
    ------------------------
    - During market stress (volatility clustering)
    - For illiquid assets (longer liquidation time)
    - When returns exhibit autocorrelation

    Parameters:
    -----------
    var_1d : float
        1-day VaR (negative number)
    holding_period : int
        Target holding period in days

    Returns:
    --------
    float
        Scaled VaR for the specified holding period
    """
    # Square-root-of-time scaling
    # Variance scales linearly with time: Var_T = T × Var_1d
    # Standard deviation scales with √T: σ_T = √T × σ_1d
    # VaR scales the same way as standard deviation
    return var_1d * np.sqrt(holding_period)


# =============================================================================
# COMPONENT VAR ANALYSIS (RISK DECOMPOSITION)
# =============================================================================

def marginal_var(weights, cov_matrix, confidence=0.99):
    """
    Calculate Marginal VaR for each asset.

    Marginal VaR measures the sensitivity of portfolio VaR to small changes
    in each asset's weight. It answers: "How much does portfolio VaR change
    if I increase this asset's weight by a small amount?"

    Mathematical Formula:
    ---------------------
    MVaR_i = z_α × (Σ × w)_i / σ_portfolio

    where:
        z_α = z-score at confidence level
        Σ = covariance matrix
        w = weight vector
        σ_portfolio = portfolio standard deviation

    Interpretation:
    ---------------
    - Positive MVaR: Adding more of this asset increases portfolio risk
    - Higher MVaR: Asset contributes more to portfolio risk per unit weight

    Use Case:
    ---------
    Risk budgeting and portfolio optimization - identify which assets
    have the largest marginal impact on portfolio risk.

    Parameters:
    -----------
    weights : array-like
        Portfolio weights (should sum to 1)
    cov_matrix : pd.DataFrame or array
        Covariance matrix of asset returns
    confidence : float
        Confidence level

    Returns:
    --------
    array
        Marginal VaR for each asset
    """
    weights = np.array(weights)
    if isinstance(cov_matrix, pd.DataFrame):
        cov_matrix = cov_matrix.values

    # Portfolio variance = w' × Σ × w
    port_var = weights @ cov_matrix @ weights

    # Portfolio volatility (standard deviation)
    port_vol = np.sqrt(port_var)

    # Z-score for confidence level (negative for left tail)
    z_score = norm.ppf(1 - confidence)

    # Marginal VaR = z × (covariance with portfolio) / portfolio vol
    # (Σ × w) gives the covariance of each asset with the portfolio
    return z_score * (cov_matrix @ weights) / port_vol


def component_var(weights, marginal_var):
    """
    Calculate Component VaR for each asset.

    Component VaR decomposes total portfolio VaR into contributions from
    each asset. This is a COMPLETE decomposition - component VaRs sum
    exactly to portfolio VaR (Euler allocation property).

    Mathematical Formula:
    ---------------------
    CVaR_i = w_i × MVaR_i

    Key Property (Euler Decomposition):
    -----------------------------------
    Σ CVaR_i = Portfolio VaR

    This additive property makes Component VaR ideal for:
    - Risk attribution (which assets contribute most to total risk?)
    - Risk budgeting (setting risk limits by asset/desk/strategy)
    - Performance attribution (risk-adjusted returns by component)

    Interpretation:
    ---------------
    - CVaR_i > 0: Asset i contributes positively to portfolio risk
    - CVaR_i < 0: Asset i provides diversification (reduces portfolio risk)
    - |CVaR_i| / Σ|CVaR_j|: Percentage contribution to total risk

    Parameters:
    -----------
    weights : array-like
        Portfolio weights
    marginal_var : array-like
        Marginal VaR for each asset (from marginal_var function)

    Returns:
    --------
    array
        Component VaR for each asset (sums to portfolio VaR)
    """
    # Component VaR = weight × Marginal VaR
    # This decomposes portfolio VaR into individual asset contributions
    return np.array(weights) * np.array(marginal_var)
