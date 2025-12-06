"""
Visualization Utilities for Market Risk VaR Analysis
=====================================================

This module provides standardized plotting functions for visualizing
Value-at-Risk (VaR) analysis results, designed for risk reporting
and regulatory presentations.

Visualization Types:
--------------------
1. Distribution Plots - Return distributions with VaR/ES thresholds
2. Backtesting Charts - Time series with violations highlighted
3. Rolling VaR Plots - Dynamic risk evolution over time
4. Component VaR Charts - Risk attribution (bar and pie)
5. Stress Test Dashboards - Scenario comparison visualizations
6. Correlation Heatmaps - Asset dependency structure

Design Principles:
------------------
- Consistent styling across all plots (seaborn-whitegrid theme)
- Clear labeling for regulatory/executive presentations
- High-quality output suitable for reports (300 DPI)
- Color schemes that work in print and digital formats
- Risk thresholds visually emphasized (red for danger)

Usage Notes:
------------
- All functions return (fig, ax) tuple for further customization
- Figures can be saved using: fig.savefig('path.png', dpi=300, bbox_inches='tight')
- Default figure sizes optimized for PowerPoint/reports

Color Conventions:
------------------
- Blue/Steelblue: Actual returns, portfolio data
- Red: VaR thresholds, violations, risk
- Dark Red: Expected Shortfall (more severe than VaR)
- Green: Safe/acceptable (e.g., returns above VaR)
- Color gradients: Risk contribution magnitude
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


# =============================================================================
# DISTRIBUTION VISUALIZATION
# =============================================================================

def plot_var_distribution(returns, var, es=None, title='Return Distribution with VaR'):
    """
    Plot return distribution with VaR (and optionally ES) threshold.

    This visualization shows:
    - Histogram of historical returns (empirical distribution)
    - Normal distribution overlay (for comparison)
    - VaR threshold line (red dashed)
    - ES threshold line (dark red solid, if provided)

    Interpretation:
    ---------------
    - Area to the left of VaR line = probability of exceeding VaR
    - Gap between distribution and normal fit shows fat tails
    - ES being further left than VaR shows tail severity

    Use Cases:
    ----------
    - Explaining VaR concept to stakeholders
    - Comparing parametric (normal) vs actual distribution
    - Visualizing tail risk (area beyond VaR)
    - Regulatory presentations

    Parameters:
    -----------
    returns : array-like
        Historical return data (typically 1-3 years of daily returns)
    var : float
        VaR threshold (negative number, e.g., -0.0441 for 4.41% loss)
    es : float, optional
        Expected Shortfall threshold (more negative than VaR)
    title : str
        Plot title

    Returns:
    --------
    tuple
        (fig, ax) matplotlib objects for further customization
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Histogram of actual returns
    # density=True normalizes so area sums to 1 (comparable to PDF)
    ax.hist(returns, bins=50, density=True, alpha=0.7, color='steelblue',
            edgecolor='black', label='Return Distribution')

    # Normal distribution overlay for comparison
    # Shows how actual returns deviate from normality assumption
    mu, sigma = np.mean(returns), np.std(returns)
    x = np.linspace(min(returns), max(returns), 100)
    ax.plot(x, norm.pdf(x, mu, sigma), 'k-', linewidth=2, label='Normal Fit')

    # VaR threshold line (vertical line at VaR value)
    # Red dashed line is standard for risk thresholds
    ax.axvline(var, color='red', linestyle='--', linewidth=2,
               label=f'VaR: {abs(var):.2%}')

    # ES threshold line if provided (further into left tail)
    if es is not None:
        ax.axvline(es, color='darkred', linestyle='-', linewidth=2,
                   label=f'ES: {abs(es):.2%}')

    # Formatting
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Return')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


# =============================================================================
# BACKTESTING VISUALIZATION
# =============================================================================

def plot_backtest(returns, var_threshold, title='VaR Backtesting'):
    """
    Plot backtesting results with violations highlighted.

    This visualization shows:
    - Time series of actual returns (blue line)
    - VaR threshold (red dashed line or rolling VaR)
    - Violations marked as red dots (returns below VaR)
    - Count of violations in legend

    Interpretation:
    ---------------
    - Violations clustered together → Model misses volatility clustering
    - Violations during specific periods → Model underestimates crisis risk
    - Violation count relative to expected → Model accuracy assessment

    Basel Context:
    --------------
    - 99% VaR over 250 days should have ~2.5 violations
    - 0-4 violations: Green zone (model accepted)
    - 5-9 violations: Yellow zone (monitoring required)
    - 10+ violations: Red zone (model rejected)

    Parameters:
    -----------
    returns : pd.Series
        Actual portfolio returns with DatetimeIndex
    var_threshold : float or pd.Series
        VaR threshold(s):
        - float: Static VaR (horizontal line)
        - pd.Series: Rolling VaR (time-varying line)
    title : str
        Plot title

    Returns:
    --------
    tuple
        (fig, ax) matplotlib objects
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot actual returns as time series
    ax.plot(returns.index, returns.values, color='blue', linewidth=0.8,
            label='Actual Returns')

    # Handle static vs rolling VaR threshold
    if isinstance(var_threshold, (int, float)):
        # Static VaR: horizontal line
        ax.axhline(var_threshold, color='red', linestyle='--', linewidth=2,
                   label=f'VaR: {abs(var_threshold):.2%}')
        violations = returns < var_threshold
    else:
        # Rolling VaR: time-varying line
        ax.plot(var_threshold.index, var_threshold.values, color='red',
                linestyle='--', linewidth=2, label='Rolling VaR')
        violations = returns < var_threshold

    # Mark violations (breaches) as red dots
    # These are the days where actual loss exceeded VaR estimate
    ax.scatter(returns.index[violations], returns[violations],
               color='red', s=50, zorder=5, label=f'Violations: {violations.sum()}')

    # Formatting
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Return')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_rolling_var(returns, rolling_var, title='Rolling VaR'):
    """
    Plot returns with rolling VaR and shaded violation regions.

    This visualization enhances the backtest plot by:
    - Shading regions where returns fell below rolling VaR
    - Showing how VaR adapts to market conditions over time
    - Highlighting periods of heightened risk

    Dynamic VaR Insight:
    --------------------
    - VaR moves lower (more negative) during volatile periods
    - VaR moves higher (less negative) during calm periods
    - Lag in VaR response shows model responsiveness

    Parameters:
    -----------
    returns : pd.Series
        Actual returns with DatetimeIndex
    rolling_var : pd.Series
        Rolling VaR values (same index as returns)
    title : str
        Plot title

    Returns:
    --------
    tuple
        (fig, ax) matplotlib objects
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot actual returns
    ax.plot(returns.index, returns.values, color='blue', linewidth=0.8,
            label='Actual Returns')

    # Plot rolling VaR
    ax.plot(rolling_var.index, rolling_var.values, color='red',
            linewidth=1.5, label='Rolling VaR')

    # Identify violation periods
    violations = returns < rolling_var

    # Shade violation regions (returns below VaR)
    # fill_between creates shaded area between two lines where condition is True
    ax.fill_between(returns.index, rolling_var.values, returns.values,
                    where=violations, color='red', alpha=0.3)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Return')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


# =============================================================================
# RISK DECOMPOSITION VISUALIZATION
# =============================================================================

def plot_component_var(component_vars, labels, title='Component VaR'):
    """
    Plot component VaR breakdown showing risk contribution by asset.

    This creates a two-panel visualization:
    - Left panel: Bar chart showing absolute Component VaR values
    - Right panel: Pie chart showing percentage contribution to total risk

    Risk Attribution Insight:
    -------------------------
    - Higher Component VaR = larger contribution to portfolio risk
    - Sum of Component VaRs = Total Portfolio VaR (Euler decomposition)
    - Can identify concentration risk and diversification benefits

    Use Cases:
    ----------
    - Risk budgeting: Allocate risk limits by asset/desk
    - Portfolio optimization: Identify high-risk contributors
    - Regulatory reporting: Demonstrate risk decomposition
    - Management dashboards: Executive risk summary

    Parameters:
    -----------
    component_vars : array-like
        Component VaR values for each asset (negative numbers)
    labels : list
        Asset labels/names (e.g., ['AAPL', 'MSFT', 'NVDA'])
    title : str
        Plot title

    Returns:
    --------
    tuple
        (fig, axes) - axes is array of two Axes objects
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Color gradient from light to dark red based on risk magnitude
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(labels)))

    # Left panel: Bar chart of absolute Component VaR
    # Convert to positive percentages for easier interpretation
    axes[0].bar(labels, np.abs(component_vars) * 100, color=colors, edgecolor='black')
    axes[0].set_ylabel('Component VaR (%)')
    axes[0].set_title(title, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Right panel: Pie chart showing relative contribution
    # Shows percentage each asset contributes to total portfolio risk
    axes[1].pie(np.abs(component_vars), labels=labels, autopct='%1.1f%%', colors=colors)
    axes[1].set_title('Risk Contribution', fontweight='bold')

    plt.tight_layout()
    return fig, axes


# =============================================================================
# STRESS TESTING VISUALIZATION
# =============================================================================

def plot_stress_scenarios(scenarios_df, title='Stress Test Results'):
    """
    Plot stress scenario results showing portfolio impact under extreme conditions.

    This creates a horizontal bar chart showing:
    - Portfolio loss for each stress scenario
    - Color-coded severity (red > 10%, orange > 5%, green < 5%)
    - Scenarios sorted or grouped by type

    Stress Test Context:
    --------------------
    Historical scenarios: Based on actual crisis periods (e.g., COVID crash)
    Hypothetical scenarios: Constructed worst-case situations (e.g., tech crash)

    Regulatory requirement: Basel III/IV requires stress testing for capital adequacy

    Color Coding:
    -------------
    - Dark Red: Severe loss (>10%)
    - Orange: Moderate loss (5-10%)
    - Green: Manageable loss (<5%)

    Parameters:
    -----------
    scenarios_df : pd.DataFrame
        DataFrame with columns:
        - 'Scenario': Scenario name/description
        - 'Portfolio Return': Portfolio loss (negative number)
    title : str
        Plot title

    Returns:
    --------
    tuple
        (fig, ax) matplotlib objects
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    scenarios = scenarios_df['Scenario'].values
    # Convert to positive percentages for display
    losses = np.abs(scenarios_df['Portfolio Return'].values) * 100

    # Color-code by severity
    # Red: severe (>10%), Orange: moderate (5-10%), Green: manageable (<5%)
    colors = ['darkred' if l > 10 else 'orange' if l > 5 else 'green' for l in losses]

    # Horizontal bar chart (barh) for better scenario label readability
    ax.barh(scenarios, losses, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Portfolio Loss (%)')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


# =============================================================================
# CORRELATION VISUALIZATION
# =============================================================================

def plot_correlation_matrix(corr_matrix, title='Correlation Matrix'):
    """
    Plot correlation heatmap showing asset dependencies.

    This visualization shows:
    - Pairwise correlations between all assets
    - Color intensity indicating correlation strength
    - Numerical values annotated in cells

    Interpretation for Risk:
    ------------------------
    - High correlation: Less diversification benefit
    - Low/negative correlation: Better diversification
    - Tech stocks often highly correlated (concentration risk)

    Color Scheme:
    -------------
    - Red: Strong positive correlation (assets move together)
    - Blue: Strong negative correlation (assets move opposite)
    - White: Near-zero correlation (independent movement)

    Use Cases:
    ----------
    - Portfolio construction: Identify diversification opportunities
    - Risk assessment: Detect concentration risk
    - Factor analysis: Understand common risk drivers
    - Regime detection: Correlations often increase during crises

    Parameters:
    -----------
    corr_matrix : pd.DataFrame
        Correlation matrix (symmetric, diagonal = 1)
    title : str
        Plot title

    Returns:
    --------
    tuple
        (fig, ax) matplotlib objects
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create upper triangle mask (optional - currently not applied)
    # This would hide redundant upper triangle since matrix is symmetric
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    # Seaborn heatmap with annotations
    # - annot=True: Show correlation values in cells
    # - fmt='.2f': Two decimal places
    # - cmap='RdYlBu_r': Red-Yellow-Blue diverging colormap (reversed)
    # - center=0: White color at correlation = 0
    # - square=True: Square cells for better aesthetics
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r',
                center=0, square=True, linewidths=1, cbar_kws={'shrink': 0.8},
                ax=ax)

    ax.set_title(title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    return fig, ax
