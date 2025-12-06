"""
Data Loading Utilities for Market Risk VaR Analysis
====================================================

This module provides functions for fetching, processing, and preparing
financial market data for Value-at-Risk (VaR) analysis.

Functionality:
--------------
1. Download price data from Yahoo Finance (via yfinance)
2. Compute returns (log and simple)
3. Calculate portfolio returns from asset weights
4. Load portfolio configuration from JSON
5. Generate summary statistics

Data Conventions:
-----------------
- Prices: Daily adjusted close prices (accounts for dividends and splits)
- Returns: Log returns are preferred for VaR analysis because:
  * Log returns are time-additive (multi-day return = sum of daily returns)
  * Log returns are symmetric (-50% and +100% have same magnitude)
  * More appropriate for continuous-time finance models
  * Better statistical properties (closer to normal distribution)

- Simple returns: Used for performance reporting and easier interpretation
  * Simple return = (P_t - P_{t-1}) / P_{t-1}
  * Not time-additive, but directly interpretable as percentage gain/loss

Date Handling:
--------------
- Start/end dates are inclusive
- Data is automatically aligned to common trading days
- Missing values (holidays, data gaps) are handled by yfinance

References:
-----------
- Yahoo Finance API via yfinance package
- Hull, J.C. "Options, Futures, and Other Derivatives" (return conventions)
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
from pathlib import Path


# =============================================================================
# PRICE DATA RETRIEVAL
# =============================================================================

def get_price_data(tickers, start_date, end_date):
    """
    Download adjusted close prices from Yahoo Finance.

    This function handles the varying column structures that yfinance returns
    depending on the number of tickers requested and the library version.

    Data Source:
    ------------
    Yahoo Finance provides:
    - Historical prices going back to IPO date
    - Adjusted close prices (adjusted for dividends and splits)
    - Global coverage (US, international markets)

    Why Adjusted Close:
    -------------------
    Using adjusted close prices ensures that:
    - Stock splits don't create artificial price jumps
    - Dividends are reinvested (total return perspective)
    - Historical returns reflect actual investor experience

    API Notes:
    ----------
    - auto_adjust=True: Returns adjusted prices directly in 'Close' column
    - progress=False: Suppresses download progress bar for cleaner output
    - yfinance may return MultiIndex columns for multiple tickers

    Parameters:
    -----------
    tickers : list
        List of stock ticker symbols (e.g., ['AAPL', 'MSFT', 'NVDA'])
        Must be valid Yahoo Finance symbols
    start_date : str
        Start date in 'YYYY-MM-DD' format (inclusive)
    end_date : str
        End date in 'YYYY-MM-DD' format (inclusive)

    Returns:
    --------
    pd.DataFrame
        DataFrame with:
        - Index: DatetimeIndex (trading days)
        - Columns: Ticker symbols
        - Values: Adjusted close prices

    Example:
    --------
    >>> prices = get_price_data(['AAPL', 'MSFT'], '2020-01-01', '2020-12-31')
    >>> prices.head()
                    AAPL       MSFT
    Date
    2020-01-02    75.09     160.62
    2020-01-03    74.36     158.62
    ...
    """
    import warnings
    warnings.filterwarnings('ignore')

    # Download data from Yahoo Finance
    # auto_adjust=True: Returns adjusted prices (for splits/dividends)
    # progress=False: Suppress progress bar
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    # Handle yfinance column structure variations
    # yfinance returns MultiIndex columns when downloading multiple tickers
    # Structure: (Price Type, Ticker) -> e.g., ('Close', 'AAPL')
    if isinstance(data.columns, pd.MultiIndex):
        # New yfinance format: MultiIndex with ('Price', 'Ticker')
        # Extract just the 'Close' prices
        prices = data['Close']
    else:
        # Single ticker or old format: columns are price types directly
        if 'Close' in data.columns:
            prices = data['Close']
        elif 'Adj Close' in data.columns:
            prices = data['Adj Close']
        else:
            # Fallback: return all data (might be single ticker)
            prices = data

    # Ensure tickers are in the requested order
    # yfinance sometimes returns tickers alphabetically
    if isinstance(prices, pd.DataFrame) and len(tickers) > 1:
        prices = prices[tickers]

    return prices


# =============================================================================
# RETURN CALCULATIONS
# =============================================================================

def compute_log_returns(prices):
    """
    Compute log returns (continuously compounded returns) from price data.

    Mathematical Formula:
    ---------------------
    r_t = ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})

    Properties of Log Returns:
    --------------------------
    1. Time Additivity:
       Multi-day return = r_1 + r_2 + ... + r_n
       This makes log returns ideal for aggregating across time

    2. Symmetry:
       -50% and +100% simple returns have same magnitude in log space
       ln(0.5) ≈ -0.693, ln(2.0) ≈ +0.693

    3. Statistical Properties:
       - More normally distributed than simple returns
       - No lower bound at -100% (theoretical)
       - Better for statistical modeling and VaR

    4. Continuous Compounding:
       P_t = P_0 × e^(r_1 + r_2 + ... + r_t)

    When to Use Log Returns:
    ------------------------
    - VaR calculations (this module)
    - Portfolio optimization
    - Time series modeling (GARCH)
    - Academic research

    When to Use Simple Returns:
    ---------------------------
    - Performance reporting to clients
    - When interpretability is priority
    - Short holding periods (difference is minimal)

    Parameters:
    -----------
    prices : pd.DataFrame or pd.Series
        Price data (adjusted close prices recommended)
        Index should be dates

    Returns:
    --------
    pd.DataFrame or pd.Series
        Log returns with first row dropped (NaN from shift)
        Same structure as input
    """
    # Log return formula: ln(P_t / P_{t-1})
    # Using np.log for natural logarithm
    # prices.shift(1) gives P_{t-1}
    # .dropna() removes the first row (NaN because no P_{t-1} exists)
    return np.log(prices / prices.shift(1)).dropna()


def compute_simple_returns(prices):
    """
    Compute simple returns (arithmetic returns) from price data.

    Mathematical Formula:
    ---------------------
    r_t = (P_t - P_{t-1}) / P_{t-1} = P_t/P_{t-1} - 1

    Properties of Simple Returns:
    -----------------------------
    1. Direct Interpretation:
       A return of 0.05 means the asset gained 5% in value

    2. NOT Time Additive:
       Two consecutive 10% gains: (1.1)(1.1) = 1.21 = 21% total, not 20%
       For multi-period, must use: (1+r_1)(1+r_2)...(1+r_n) - 1

    3. Bounded Below:
       Cannot be less than -100% (can't lose more than investment)

    4. Cross-Sectional Additivity:
       Portfolio return = Σ w_i × r_i (weighted sum of asset returns)
       This is why simple returns are used for portfolio calculations

    Relationship to Log Returns:
    ----------------------------
    For small returns, log return ≈ simple return
    Exact relationship: r_simple = e^(r_log) - 1
                       r_log = ln(1 + r_simple)

    Parameters:
    -----------
    prices : pd.DataFrame or pd.Series
        Price data

    Returns:
    --------
    pd.DataFrame or pd.Series
        Simple returns (percentage change)
    """
    # pct_change() computes (P_t - P_{t-1}) / P_{t-1}
    # .dropna() removes the first row (NaN)
    return prices.pct_change().dropna()


# =============================================================================
# PORTFOLIO RETURN CALCULATION
# =============================================================================

def compute_portfolio_returns(asset_returns, weights, normalize=True):
    """
    Compute portfolio returns as weighted sum of asset returns.

    Mathematical Formula:
    ---------------------
    R_portfolio = Σ w_i × R_i = w' × R

    where:
        w_i = weight of asset i (fraction of portfolio value)
        R_i = return of asset i
        w' = weight vector transposed

    Weight Normalization:
    ---------------------
    If normalize=True, weights are scaled to sum to 1:
        w_normalized = w / Σw

    This ensures we're computing returns for a fully invested portfolio.

    Cross-Sectional Aggregation Note:
    ---------------------------------
    This formula works for SIMPLE returns (cross-sectionally additive).
    For LOG returns, this is an approximation that works well when:
    - Returns are small (daily returns)
    - Weights don't change significantly

    For exact log portfolio returns, use:
        R_log_portfolio = ln(Σ w_i × e^(R_log_i))

    But for daily returns, the approximation error is negligible (<0.01%).

    Parameters:
    -----------
    asset_returns : pd.DataFrame
        Returns for each asset (columns = tickers, rows = dates)
    weights : array-like
        Portfolio weights for each asset
        Should be in same order as columns in asset_returns
    normalize : bool
        If True, normalize weights to sum to 1 (default True)

    Returns:
    --------
    pd.Series
        Portfolio returns (same index as asset_returns)
        Single column of weighted average returns

    Example:
    --------
    >>> returns = pd.DataFrame({
    ...     'AAPL': [0.01, -0.02, 0.03],
    ...     'MSFT': [0.02, -0.01, 0.01]
    ... })
    >>> weights = [0.6, 0.4]  # 60% AAPL, 40% MSFT
    >>> port_returns = compute_portfolio_returns(returns, weights)
    """
    # Convert weights to numpy array for matrix operations
    weights = np.array(weights)

    # Normalize weights to sum to 1 (fully invested portfolio)
    if normalize:
        weights = weights / weights.sum()

    # Matrix multiplication: returns × weights
    # For each day: sum of (weight_i × return_i)
    return asset_returns.dot(weights)


# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

def load_config(config_path='portfolio_config.json'):
    """
    Load portfolio configuration from JSON file.

    The configuration file contains all parameters needed for the analysis:
    - Portfolio composition (tickers, weights, names)
    - Date ranges (training, testing periods)
    - VaR parameters (confidence levels, simulations)
    - Risk factors for factor model
    - Stress test scenarios

    Configuration File Structure:
    -----------------------------
    {
        "portfolio": {
            "tickers": ["AAPL", "MSFT", ...],
            "weights": [0.30, 0.25, ...],
            "names": ["Apple Inc.", "Microsoft Corp.", ...]
        },
        "dates": {
            "start": "2022-01-01",
            "end": "2025-12-03",
            "train_start": "2022-01-01",
            "train_end": "2024-11-30",
            "test_start": "2024-12-01",
            "test_end": "2025-12-03"
        },
        "var_params": {
            "confidence_var": 0.99,
            "confidence_es": 0.975,
            "simulations": 10000
        },
        ...
    }

    Parameters:
    -----------
    config_path : str
        Path to JSON configuration file (default: 'portfolio_config.json')
        Can be absolute or relative path

    Returns:
    --------
    dict
        Configuration dictionary with all parameters
    """
    with open(config_path, 'r') as f:
        return json.load(f)


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def get_summary_statistics(returns):
    """
    Compute summary statistics for return data.

    Calculates key descriptive statistics used in risk analysis:
    - Central tendency: mean (daily and annualized)
    - Dispersion: standard deviation (daily and annualized)
    - Extremes: minimum and maximum returns
    - Distribution shape: skewness and kurtosis

    Annualization:
    --------------
    - Mean: Daily mean × 252 trading days
    - Std Dev: Daily std × √252 (variance scales linearly with time)

    Skewness Interpretation:
    ------------------------
    - Skew < 0: Left tail heavier (more extreme losses)
    - Skew = 0: Symmetric distribution
    - Skew > 0: Right tail heavier (more extreme gains)
    - Financial returns typically show negative skewness

    Kurtosis Interpretation:
    ------------------------
    - Kurt = 0: Normal distribution (mesokurtic)
    - Kurt > 0: Fat tails (leptokurtic) - more extreme events
    - Kurt < 0: Thin tails (platykurtic) - fewer extreme events
    - Financial returns typically show positive excess kurtosis

    Note: This uses Fisher's definition of kurtosis (excess kurtosis)
    where normal distribution has kurtosis = 0.

    Parameters:
    -----------
    returns : pd.Series or pd.DataFrame
        Return data (log returns or simple returns)

    Returns:
    --------
    pd.DataFrame
        Summary statistics with metrics as rows and assets as columns
    """
    from scipy.stats import skew, kurtosis

    # Convert Series to DataFrame for consistent handling
    if isinstance(returns, pd.Series):
        returns = returns.to_frame('Returns')

    # Compute all statistics
    stats = pd.DataFrame({
        'Mean (Daily)': returns.mean(),
        'Mean (Annual)': returns.mean() * 252,  # Annualize: 252 trading days
        'Std (Daily)': returns.std(),
        'Std (Annual)': returns.std() * np.sqrt(252),  # √252 scaling
        'Min': returns.min(),
        'Max': returns.max(),
        'Skewness': returns.apply(skew),  # scipy.stats.skew
        'Kurtosis': returns.apply(kurtosis)  # Fisher's definition (excess)
    })

    # Transpose so metrics are rows (better for display)
    return stats.T
