"""
Backtesting Functions for VaR Model Validation
===============================================

This module implements regulatory-compliant backtesting methodologies for
validating Value-at-Risk (VaR) models according to Basel III/IV requirements.

Backtesting Framework:
----------------------
1. Basel III Traffic Light System (Green/Yellow/Red zones)
2. Kupiec Test (Proportion of Failures / POF Test)
3. Christoffersen Test (Independence of Violations)
4. Rolling Backtests (time-varying VaR validation)

Regulatory Context:
-------------------
Basel Committee on Banking Supervision (BCBS) requires banks to backtest
their VaR models using a 250-day rolling window. Models are classified into
three zones based on the number of VaR violations:

- Green Zone (0-4 violations): Model accepted
- Yellow Zone (5-9 violations): Model requires monitoring, capital surcharge
- Red Zone (10+ violations): Model rejected, significant capital penalty

Key Concepts:
-------------
- Violation: When actual loss exceeds VaR estimate
- Expected violations at 99% confidence over 250 days: 2.5 (= 250 × 0.01)
- Type I error: Rejecting a good model (false positive)
- Type II error: Accepting a bad model (false negative)

References:
-----------
- BCBS (1996). "Supervisory Framework for the Use of Backtesting"
- Christoffersen, P. (1998). "Evaluating Interval Forecasts"
- Kupiec, P. (1995). "Techniques for Verifying the Accuracy of Risk Models"
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm


# =============================================================================
# BASEL III TRAFFIC LIGHT SYSTEM
# =============================================================================

def basel_zone(violations):
    """
    Determine Basel III traffic light zone based on number of violations.

    The Basel Committee established a traffic light approach for backtesting
    VaR models. Over a 250-day period at 99% confidence, the expected number
    of violations is 2.5. The zones account for sampling variation.

    Zone Classification:
    --------------------
    GREEN (0-4 violations):
        - Model accepted
        - No capital surcharge
        - Probability of 4 or fewer violations if model is correct: ~89%

    YELLOW (5-9 violations):
        - Model requires monitoring
        - Progressive capital multiplier increases
        - May indicate model issues or period of unusual market stress

    RED (10+ violations):
        - Model rejected
        - Maximum capital surcharge (100% increase)
        - Model must be reviewed and potentially replaced

    Capital Multiplier Schedule (Basel II/III):
    -------------------------------------------
    Violations | Zone   | Multiplier Increase
    0-4        | Green  | 0.00 (base = 3.0)
    5          | Yellow | 0.40 (factor = 3.4)
    6          | Yellow | 0.50 (factor = 3.5)
    7          | Yellow | 0.65 (factor = 3.65)
    8          | Yellow | 0.75 (factor = 3.75)
    9          | Yellow | 0.85 (factor = 3.85)
    10+        | Red    | 1.00 (factor = 4.0)

    Parameters:
    -----------
    violations : int
        Number of VaR violations in the backtesting period

    Returns:
    --------
    tuple
        (zone_name, color, multiplier_increase)
        - zone_name: 'Green', 'Yellow', or 'Red'
        - color: 'green', 'yellow', or 'red' (for visualization)
        - multiplier_increase: addition to base capital multiplier (0.0-1.0)
    """
    if violations <= 4:
        # Green zone: Model is acceptable
        # At 99% confidence, getting 0-4 violations in 250 days
        # has approximately 89% probability under the null hypothesis
        return ('Green', 'green', 0.00)
    elif violations <= 9:
        # Yellow zone: Model requires monitoring
        # Capital multiplier increases progressively
        multipliers = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}
        return ('Yellow', 'yellow', multipliers.get(violations, 0.85))
    else:
        # Red zone: Model is rejected
        # Maximum capital penalty applied
        return ('Red', 'red', 1.00)


# =============================================================================
# STANDARD VAR BACKTESTING
# =============================================================================

def backtest_var(actual_returns, var_threshold, confidence=0.99):
    """
    Perform VaR backtesting - count violations and assess model quality.

    A violation occurs when the actual return is WORSE than the VaR estimate:
        actual_return < VaR_threshold

    This function supports both:
    - Static backtesting: Single VaR threshold applied to all days
    - Dynamic backtesting: Rolling VaR threshold that changes daily

    Process:
    --------
    1. Compare each actual return to corresponding VaR threshold
    2. Count violations (days where loss exceeded VaR)
    3. Calculate violation rate
    4. Determine Basel zone classification

    Key Metrics:
    ------------
    - violations: Raw count of VaR breaches
    - violation_rate: violations / total_days (should be close to 1-confidence)
    - expected_rate: 1 - confidence (e.g., 1% for 99% VaR)

    Parameters:
    -----------
    actual_returns : pd.Series
        Actual portfolio returns (realized P&L)
    var_threshold : float or pd.Series
        VaR threshold(s):
        - float: Static VaR (same threshold every day)
        - pd.Series: Rolling VaR (daily updated threshold)
    confidence : float
        VaR confidence level (e.g., 0.99 for 99%)

    Returns:
    --------
    dict
        Backtesting results including:
        - violations: Number of VaR breaches
        - total_days: Length of test period
        - violation_rate: Observed violation frequency
        - expected_rate: Expected violation frequency (1-confidence)
        - zone: Basel traffic light zone
        - color: Zone color for visualization
        - multiplier_increase: Capital multiplier adjustment
    """
    # Handle static vs dynamic VaR thresholds
    if isinstance(var_threshold, (int, float)):
        # Static VaR: compare all returns to single threshold
        violations = actual_returns < var_threshold
    else:
        # Dynamic VaR: align indices and compare element-wise
        common_idx = actual_returns.index.intersection(var_threshold.index)
        actual_returns = actual_returns.loc[common_idx]
        var_threshold = var_threshold.loc[common_idx]
        violations = actual_returns < var_threshold

    # Count and calculate metrics
    num_violations = violations.sum()
    total_days = len(actual_returns)
    violation_rate = num_violations / total_days
    expected_rate = 1 - confidence  # e.g., 0.01 for 99% confidence

    # Get Basel zone classification
    zone, color, multiplier = basel_zone(num_violations)

    return {
        'violations': int(num_violations),
        'total_days': total_days,
        'violation_rate': violation_rate,
        'expected_rate': expected_rate,
        'zone': zone,
        'color': color,
        'multiplier_increase': multiplier
    }


# =============================================================================
# KUPIEC TEST (PROPORTION OF FAILURES)
# =============================================================================

def kupiec_test(violations, total_days, confidence=0.99):
    """
    Perform Kupiec POF (Proportion of Failures) test.

    The Kupiec test is a likelihood ratio test that evaluates whether the
    observed violation rate is statistically consistent with the expected
    violation rate under the VaR model.

    Hypothesis:
    -----------
    H0: p = α (observed violation rate equals expected rate)
    H1: p ≠ α (violation rate differs from expected)

    where α = 1 - confidence (e.g., 0.01 for 99% VaR)

    Test Statistic:
    ---------------
    LR_POF = -2 × ln[(1-α)^(T-x) × α^x / (1-p̂)^(T-x) × p̂^x]

    where:
        T = total number of observations
        x = number of violations
        α = expected violation rate (1 - confidence)
        p̂ = observed violation rate (x/T)

    Distribution:
    -------------
    Under H0, LR_POF ~ χ²(1) (chi-squared with 1 degree of freedom)

    Decision Rule:
    --------------
    - If p-value > 0.05: Fail to reject H0 (model accepted)
    - If p-value ≤ 0.05: Reject H0 (model rejected)

    Interpretation:
    ---------------
    - Low p-value: Violation rate significantly different from expected
    - High p-value: Violation rate consistent with VaR model assumptions

    Limitations:
    ------------
    - Only tests UNCONDITIONAL coverage (correct long-run proportion)
    - Does NOT test independence (clustering of violations)
    - Use with Christoffersen test for complete validation

    Parameters:
    -----------
    violations : int
        Number of VaR violations observed
    total_days : int
        Total number of observations in backtest period
    confidence : float
        VaR confidence level

    Returns:
    --------
    dict
        Test results including:
        - lr_statistic: Likelihood ratio test statistic
        - p_value: P-value from chi-squared distribution
        - critical_value: 95% critical value (3.841)
        - reject_H0: Boolean indicating if model is rejected
        - conclusion: Text summary of test result
    """
    # Expected violation rate (alpha)
    alpha = 1 - confidence  # e.g., 0.01 for 99% confidence

    x = violations  # Number of violations
    T = total_days  # Total observations

    # Observed violation rate (with numerical safeguards)
    p_hat = x / T if x > 0 else 0.0001

    # Handle edge cases to avoid log(0)
    if x == 0:
        p_hat = 0.0001  # Small positive number to avoid log(0)
    if x == T:
        p_hat = 0.9999  # Slightly less than 1 to avoid log(0)

    # Likelihood under H0 (expected rate = alpha)
    # L0 = (1-α)^(T-x) × α^x
    lr_num = ((1 - alpha) ** (T - x)) * (alpha ** x)

    # Likelihood under H1 (estimated rate = p_hat)
    # L1 = (1-p̂)^(T-x) × p̂^x
    lr_den = ((1 - p_hat) ** (T - x)) * (p_hat ** x)

    # Likelihood Ratio test statistic
    # LR = -2 × ln(L0 / L1)
    lr_stat = -2 * np.log(lr_num / lr_den)

    # P-value from chi-squared distribution with 1 degree of freedom
    p_value = 1 - chi2.cdf(lr_stat, df=1)

    # Critical value at 95% significance level
    # χ²(1, 0.95) = 3.841
    critical_value = chi2.ppf(0.95, df=1)

    return {
        'violations': x,
        'total_days': T,
        'expected_rate': alpha,
        'observed_rate': x / T,
        'lr_statistic': lr_stat,
        'p_value': p_value,
        'critical_value': critical_value,
        'reject_H0': lr_stat > critical_value,
        'conclusion': 'Model Rejected' if lr_stat > critical_value else 'Model Accepted'
    }


# =============================================================================
# CHRISTOFFERSEN TEST (INDEPENDENCE OF VIOLATIONS)
# =============================================================================

def christoffersen_test(violations_series):
    """
    Perform Christoffersen independence test.

    While the Kupiec test checks if the PROPORTION of violations is correct,
    the Christoffersen test checks if violations are INDEPENDENT (not clustered).

    Motivation:
    -----------
    A VaR model could have the correct average violation rate but still be
    problematic if violations cluster together (e.g., multiple violations
    in a row during market stress). This clustering suggests the model
    fails to capture volatility dynamics.

    Hypothesis:
    -----------
    H0: Violations are independent (no clustering)
    H1: Violations are dependent (clustered)

    Test Construction:
    ------------------
    Uses a first-order Markov chain to model violation sequences:

    Transition Matrix:
                    Next Day
                    No Viol | Viol
    Today No Viol |  n00   | n01   → π01 = P(Viol|No Viol)
    Today Viol    |  n10   | n11   → π11 = P(Viol|Viol)

    where:
        n00 = count of (no violation → no violation) transitions
        n01 = count of (no violation → violation) transitions
        n10 = count of (violation → no violation) transitions
        n11 = count of (violation → violation) transitions

    Under independence: π01 = π11 = π (overall violation rate)

    Test Statistic:
    ---------------
    LR_ind = -2 × ln(L0 / L1)

    where:
        L0 = likelihood under independence
        L1 = likelihood under Markov dependence

    Distribution:
    -------------
    Under H0, LR_ind ~ χ²(1)

    Interpretation:
    ---------------
    - reject_H0 = True: Violations are clustered (model issue)
    - reject_H0 = False: Violations appear independent (good model)

    Warning Signs:
    --------------
    - High π11: Violations tend to follow each other (volatility clustering)
    - π11 >> π01: Model underestimates risk during stress periods

    Parameters:
    -----------
    violations_series : pd.Series
        Boolean series where True = violation, False = no violation
        Index should be dates

    Returns:
    --------
    dict
        Test results including:
        - n00, n01, n10, n11: Transition counts
        - pi01: P(violation | no violation yesterday)
        - pi11: P(violation | violation yesterday)
        - lr_statistic: Test statistic
        - p_value: P-value
        - independent: Boolean indicating if violations are independent
    """
    # Convert to integer array (0 = no violation, 1 = violation)
    violations = violations_series.astype(int).values
    T = len(violations)

    # Count transitions between states
    # Using first-order Markov chain: look at consecutive day pairs
    n00 = n01 = n10 = n11 = 0

    for i in range(1, T):
        if violations[i - 1] == 0 and violations[i] == 0:
            n00 += 1  # No violation → No violation
        elif violations[i - 1] == 0 and violations[i] == 1:
            n01 += 1  # No violation → Violation
        elif violations[i - 1] == 1 and violations[i] == 0:
            n10 += 1  # Violation → No violation
        else:
            n11 += 1  # Violation → Violation

    # Transition probabilities
    # π01 = P(violation today | no violation yesterday)
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0

    # π11 = P(violation today | violation yesterday)
    # High π11 indicates clustering
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0

    # Overall violation probability (under independence)
    pi = (n01 + n11) / (T - 1) if T > 1 else 0

    # Small epsilon to avoid log(0)
    eps = 1e-10

    # Likelihood under H0 (independence): same probability regardless of yesterday
    # L0 = (1-π)^(n00+n10) × π^(n01+n11)
    L0 = ((1 - pi + eps) ** (n00 + n10)) * ((pi + eps) ** (n01 + n11))

    # Likelihood under H1 (Markov dependence): different probabilities
    # L1 = (1-π01)^n00 × π01^n01 × (1-π11)^n10 × π11^n11
    L1 = ((1 - pi01 + eps) ** n00) * ((pi01 + eps) ** n01) * \
         ((1 - pi11 + eps) ** n10) * ((pi11 + eps) ** n11)

    # Likelihood ratio test statistic
    lr_stat = -2 * np.log(L0 / L1) if L1 > 0 else 0

    # P-value from chi-squared distribution
    p_value = 1 - chi2.cdf(lr_stat, df=1)

    # Critical value at 95% significance
    critical_value = chi2.ppf(0.95, df=1)

    return {
        'n00': n00, 'n01': n01, 'n10': n10, 'n11': n11,
        'pi01': pi01,  # P(violation | no violation yesterday)
        'pi11': pi11,  # P(violation | violation yesterday) - clustering indicator
        'lr_statistic': lr_stat,
        'p_value': p_value,
        'critical_value': critical_value,
        'reject_H0': lr_stat > critical_value,
        'independent': lr_stat <= critical_value  # True = good (no clustering)
    }


# =============================================================================
# ROLLING VAR BACKTESTING
# =============================================================================

def rolling_var_backtest(returns, window=250, confidence=0.99, method='historical'):
    """
    Perform rolling VaR backtesting.

    This function implements dynamic backtesting where VaR is re-estimated
    each day using a rolling window of historical data. This is more
    realistic than static backtesting as it mimics actual risk management
    practice.

    Process:
    --------
    For each day t (from window to end):
    1. Use returns from [t-window, t-1] to estimate VaR
    2. Compare day t's actual return to VaR estimate
    3. Record violation if actual return < VaR

    Rolling Window Rationale:
    -------------------------
    - 250 days ≈ 1 trading year (standard for Basel backtesting)
    - Captures recent market conditions and volatility regime
    - Balances responsiveness with statistical reliability

    Methods Available:
    ------------------
    - 'historical': Non-parametric, uses percentile of rolling window
    - 'parametric': Assumes normality, uses μ + z×σ from rolling window
    - 'monte_carlo': Simulates from rolling window parameters

    Output Usage:
    -------------
    The returned DataFrame can be used to:
    - Count total violations for Kupiec test
    - Create violation series for Christoffersen test
    - Visualize VaR breaches over time
    - Analyze violations during specific market periods

    Parameters:
    -----------
    returns : pd.Series
        Full return series (must be longer than window)
    window : int
        Rolling window size (default 250 = ~1 trading year)
    confidence : float
        VaR confidence level
    method : str
        VaR calculation method:
        - 'historical': Percentile-based (recommended)
        - 'parametric': Normal distribution assumption
        - 'monte_carlo': Simulation-based

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns:
        - Actual_Return: Realized return for the day
        - Rolling_VaR: VaR estimate (calculated from prior window)
        - Violation: Boolean indicating if VaR was breached
        Index is date
    """
    results = []

    # Start from position 'window' (need 'window' days of history)
    for i in range(window, len(returns)):
        # Training window: previous 'window' days
        train = returns.iloc[i - window:i]

        # Test point: today's actual return
        actual = returns.iloc[i]
        date = returns.index[i]

        # Calculate VaR based on method
        if method == 'historical':
            # Non-parametric: use percentile of historical returns
            var = np.percentile(train, (1 - confidence) * 100)

        elif method == 'parametric':
            # Parametric: assume normality
            mu, sigma = train.mean(), train.std()
            var = mu + norm.ppf(1 - confidence) * sigma

        elif method == 'monte_carlo':
            # Monte Carlo: simulate from estimated parameters
            mu, sigma = train.mean(), train.std()
            simulated = np.random.normal(mu, sigma, 10000)
            var = np.percentile(simulated, (1 - confidence) * 100)

        # Record results
        results.append({
            'Date': date,
            'Actual_Return': actual,
            'Rolling_VaR': var,
            'Violation': actual < var  # True if loss exceeded VaR
        })

    return pd.DataFrame(results).set_index('Date')
