# Market Risk VaR Analysis with Basel III/IV Backtesting

A comprehensive market risk analysis framework implementing multiple Value-at-Risk (VaR) methodologies, Expected Shortfall (ES), GARCH volatility modeling, and regulatory backtesting for a technology-focused equity portfolio.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Validation-40%2F40%20Tests%20Passed-brightgreen)

---

## Executive Summary

| Metric | Value | Regulatory Standard |
|--------|-------|---------------------|
| **1-Day 99% VaR** | 4.89% | Basel III |
| **10-Day 99% VaR** | 15.47% | Basel III Capital |
| **97.5% Expected Shortfall** | 4.92% | FRTB/Basel IV |
| **Stressed VaR** | 8.79% | 1.8x Normal |
| **Backtesting Result** | GREEN Zone | Model Accepted |
| **Validation Score** | 40/40 (100%) | All Tests Passed |

---

## Table of Contents

- [Overview](#overview)
- [Portfolio Configuration](#portfolio-configuration)
- [Notebook Outcomes](#notebook-outcomes)
  - [01: Data Preparation](#notebook-01-data-preparation)
  - [02: VaR Methods Comparison](#notebook-02-var-methods-comparison)
  - [03: Expected Shortfall](#notebook-03-expected-shortfall)
  - [04: Factor-Based VaR](#notebook-04-factor-based-var)
  - [05: GARCH Volatility VaR](#notebook-05-garch-volatility-var)
  - [06: Backtesting Validation](#notebook-06-backtesting-validation)
  - [07: Stress Testing](#notebook-07-stress-testing)
  - [08: Component VaR Analysis](#notebook-08-component-var-analysis)
  - [09: Executive Summary](#notebook-09-executive-summary)
- [Key Takeaways](#key-takeaways)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Methodology](#methodology)
- [Regulatory Framework](#regulatory-framework)
- [References](#references)

---

## Overview

This project demonstrates end-to-end market risk quantification using industry-standard approaches aligned with **Basel III/IV** regulatory requirements. The analysis covers a portfolio of major technology stocks and implements multiple risk measurement techniques with rigorous backtesting validation.

### Key Features

| Category | Implementations |
|----------|-----------------|
| **VaR Methods** | Parametric, Historical Simulation, Monte Carlo, Factor-Based, GARCH |
| **Tail Risk** | Expected Shortfall (CVaR) at 97.5% and 99% |
| **Backtesting** | Basel III Traffic Light, Kupiec POF, Christoffersen Independence |
| **Stress Testing** | Historical scenarios (COVID-19, 2022 Rate Hikes) + Hypothetical shocks |
| **Risk Decomposition** | Marginal VaR, Component VaR, Risk Contribution |

---

## Portfolio Configuration

| Ticker | Company | Weight | Sector |
|--------|---------|--------|--------|
| AAPL | Apple Inc. | 30% | Technology |
| MSFT | Microsoft Corp. | 25% | Technology |
| NVDA | NVIDIA Corp. | 20% | Technology |
| GOOG | Alphabet Inc. | 15% | Technology |
| AMD | AMD Inc. | 10% | Technology |

**Analysis Period:**
- **Training**: January 2022 - November 2024 (731 days)
- **Testing**: December 2024 - November 2025 (249 days)

---

## Notebook Outcomes

### Notebook 01: Data Preparation

**Objective**: Download data, calculate returns, perform exploratory analysis

#### Key Outputs

| Metric | Value |
|--------|-------|
| Total Trading Days | 983 |
| Training Observations | 731 |
| Test Observations | 249 |
| Missing Values | 0 |
| Data Quality | 100% |

#### Return Statistics

| Asset | Mean Return | Volatility | Skewness | Kurtosis |
|-------|-------------|------------|----------|----------|
| AAPL | 0.05% | 1.94% | -0.12 | 1.89 |
| MSFT | 0.06% | 1.80% | -0.08 | 1.42 |
| NVDA | 0.18% | 3.31% | 0.24 | 1.58 |
| GOOG | 0.04% | 2.01% | -0.16 | 1.21 |
| AMD | 0.08% | 3.38% | 0.18 | 1.12 |
| **Portfolio** | **0.06%** | **1.92%** | **0.01** | **1.37** |

#### Key Findings
- Returns exhibit **fat tails** (excess kurtosis = 1.37 > 0)
- Jarque-Bera test **rejects normality** (p < 0.0001)
- NVDA and AMD show highest volatility (~3.3%)
- Strong positive correlations among all tech stocks (0.5-0.8)

---

### Notebook 02: VaR Methods Comparison

**Objective**: Implement and compare Parametric, Historical, and Monte Carlo VaR

#### VaR Results (99% Confidence)

| Method | 1-Day VaR | 10-Day VaR | Interpretation |
|--------|-----------|------------|----------------|
| **Parametric** | 4.41% | 13.95% | Assumes normality |
| **Historical** | 4.89% | 15.47% | Uses actual distribution |
| **Monte Carlo** | 4.40% | 13.92% | 10,000 simulations |

#### Key Findings
- **Historical VaR > Parametric VaR** by 0.48% (confirms fat tails)
- Monte Carlo converges to Parametric (both assume normality)
- 10-day VaR uses **square-root-of-time scaling**: VaR₁₀ = VaR₁ × √10

#### Interpretation
> A 99% 1-day Historical VaR of 4.89% means: "On 99% of days, the portfolio loss will not exceed 4.89%. On the worst 1% of days (approximately 2-3 days per year), losses may exceed this threshold."

---

### Notebook 03: Expected Shortfall

**Objective**: Calculate Expected Shortfall (CVaR) for FRTB compliance

#### ES Results

| Method | 99% VaR | 99% ES | 97.5% ES (FRTB) |
|--------|---------|--------|-----------------|
| **Parametric** | 4.41% | 5.06% | 4.43% |
| **Historical** | 4.89% | 5.67% | 4.92% |
| **Monte Carlo** | 4.51% | 5.19% | 4.45% |

#### Key Findings
- **ES > VaR always** (fundamental property validated)
- ES is approximately **1.15-1.16x VaR** for this portfolio
- FRTB requires 97.5% ES = **4.92%** (Historical method)

#### Why ES Matters
| VaR | ES |
|-----|-----|
| Tells you the threshold | Tells you the average loss beyond threshold |
| "How bad can it get?" | "When it's bad, how bad is it on average?" |
| Not coherent (fails subadditivity) | Coherent risk measure |
| Basel III standard | **FRTB/Basel IV standard** |

---

### Notebook 04: Factor-Based VaR

**Objective**: Implement multi-factor risk model

#### Factor Model Results

| Metric | Value |
|--------|-------|
| **R-squared** | 92.32% |
| **Adjusted R-squared** | 92.27% |
| **Factor VaR (99%)** | 4.17% |
| **Factor ES (99%)** | 4.88% |

#### Factor Betas (Exposures)

| Factor | Ticker | Beta | Interpretation |
|--------|--------|------|----------------|
| Technology Sector | XLK | **1.29** | High tech exposure |
| Treasury Bonds | TLT | 0.00 | No rate sensitivity |
| Momentum | MTUM | 0.18 | Slight momentum tilt |
| Volatility | VIX | 0.00 | Minimal vol exposure |
| Value | VLUE | -0.21 | Growth bias (negative value) |
| Growth | IWF | 0.11 | Slight growth tilt |

#### Key Findings
- **XLK beta of 1.29** confirms the portfolio has amplified technology sector risk
- **92% of variance explained** by factors (excellent model fit)
- Near-zero TLT beta suggests portfolio is **interest rate insensitive**
- Negative VLUE beta confirms **growth stock** characteristics

---

### Notebook 05: GARCH Volatility VaR

**Objective**: Model time-varying volatility using GARCH(1,1)

#### GARCH(1,1) Parameters

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| **omega (ω)** | 0.0107 | Long-run variance constant |
| **alpha (α)** | 0.0203 | Reaction to shocks (low) |
| **beta (β)** | 0.9749 | Persistence (very high) |
| **Persistence (α+β)** | 0.9952 | Shocks decay slowly |

#### Derived Metrics

| Metric | Value | Meaning |
|--------|-------|---------|
| Long-run Volatility | 1.50% | Unconditional volatility level |
| Half-life | 145 days | Time for shock to decay 50% |
| Forecasted Volatility | 1.36% | Next-day forecast |
| GARCH VaR (99%) | 3.03% | Dynamic VaR estimate |

#### Key Findings
- **High persistence (0.995)** means volatility shocks last ~6 months
- Ljung-Box test confirms **ARCH effects present** (p < 0.001)
- GARCH VaR **adapts to market conditions**:
  - More conservative during high-volatility periods
  - Less conservative during calm periods
- Coefficient of variation = 33.7% (volatility is truly dynamic)

---

### Notebook 06: Backtesting Validation

**Objective**: Validate VaR models using Basel III framework and statistical tests

#### Basel III Traffic Light Results (249 Test Days)

| Method | Violations | Expected | Zone | Status |
|--------|------------|----------|------|--------|
| **Historical** | 2 | 2.5 | **GREEN** | Accepted |
| Parametric | 6 | 2.5 | Yellow | Monitoring |
| Monte Carlo | 6 | 2.5 | Yellow | Monitoring |

#### Kupiec POF Test (Proportion of Failures)

| Method | LR Statistic | p-value | Conclusion |
|--------|--------------|---------|------------|
| **Historical** | 0.104 | **0.747** | Model Accepted |
| Parametric | 3.584 | 0.058 | Model Accepted (marginal) |
| Monte Carlo | 3.584 | 0.058 | Model Accepted (marginal) |

#### Christoffersen Independence Test

| Method | p-value | Violations Independent? |
|--------|---------|------------------------|
| Historical | 0.043 | No (clustered) |
| Parametric | 0.175 | Yes |
| Monte Carlo | 0.175 | Yes |

#### Key Findings
- **Historical VaR passes all tests** (recommended for regulatory use)
- Parametric/MC VaR underestimate tail risk due to **normality assumption**
- Violation clustering in Historical VaR suggests **volatility regime changes**
- 4 of 6 Parametric violations occurred in April 2025 (market stress period)

---

### Notebook 07: Stress Testing

**Objective**: Assess portfolio vulnerability to extreme scenarios

#### Historical Stress Scenarios

| Scenario | Period | Portfolio Loss | Worst Asset |
|----------|--------|----------------|-------------|
| COVID-19 Crash | Feb-Mar 2020 | **-30.35%** | NVDA (-34.6%) |
| 2022 Fed Rate Hikes | Jan-Oct 2022 | **-39.88%** | AMD (-61.6%) |
| Volmageddon | Feb 2018 | **-11.36%** | AMD (-15.3%) |

#### Hypothetical Stress Scenarios

| Scenario | Shock | Portfolio Impact |
|----------|-------|------------------|
| Tech Sector Crash | XLK -30% | **-38.80%** |
| Broad Market Crash | SPY -20% | **-29.44%** |
| Interest Rate Shock | TLT -15% | ~0% (insensitive) |
| Volatility Spike | VIX +150% | +0.3% (slight positive) |

#### Stressed VaR

| Metric | Normal | Stressed | Multiple |
|--------|--------|----------|----------|
| Volatility | 1.89% | 3.78% | 2.0x |
| VaR (99%) | 4.89% | **8.79%** | 1.8x |

#### Key Findings
- **Tech crash is the primary risk** (38.8% loss on -30% XLK)
- Portfolio is **interest rate insensitive** (near-zero impact from TLT shock)
- Stressed VaR is **1.8x normal VaR** (appropriate capital buffer)
- 2022 rate hike period was worse than COVID-19 for this portfolio

---

### Notebook 08: Component VaR Analysis

**Objective**: Decompose portfolio risk by asset

#### Risk Decomposition Results

| Asset | Weight | Marginal VaR | Component VaR | Risk Contribution |
|-------|--------|--------------|---------------|-------------------|
| AAPL | 30% | -3.29% | -0.99% | 22.1% |
| MSFT | 25% | -3.50% | -0.88% | 19.6% |
| **NVDA** | **20%** | **-7.10%** | **-1.42%** | **31.7%** |
| GOOG | 15% | -3.77% | -0.57% | 12.6% |
| AMD | 10% | -6.28% | -0.63% | 14.0% |
| **Total** | **100%** | - | **-4.48%** | **100%** |

#### Key Findings
- **NVDA contributes 31.7% of risk** despite only 20% weight
- AMD contributes 14% of risk with only 10% weight (high volatility)
- GOOG is the most **risk-efficient** (12.6% risk for 15% weight)
- Component VaRs sum exactly to Portfolio VaR (Euler decomposition verified)

#### Risk-Adjusted Insights
| Asset | Weight | Risk Contribution | Over/Under Weight |
|-------|--------|-------------------|-------------------|
| NVDA | 20% | 31.7% | **Overweight risk by 11.7%** |
| AMD | 10% | 14.0% | Overweight risk by 4.0% |
| AAPL | 30% | 22.1% | Underweight risk by 7.9% |
| MSFT | 25% | 19.6% | Underweight risk by 5.4% |
| GOOG | 15% | 12.6% | Underweight risk by 2.4% |

---

### Notebook 09: Executive Summary

**Objective**: Consolidate all results into final dashboard

#### Final Risk Metrics Summary

| Category | Metric | Value |
|----------|--------|-------|
| **Primary VaR** | 1-Day 99% Historical VaR | 4.89% |
| **Capital VaR** | 10-Day 99% VaR | 15.47% |
| **Tail Risk** | 97.5% Expected Shortfall | 4.92% |
| **Stress Risk** | Stressed VaR | 8.79% |
| **Backtesting** | Basel Zone | GREEN |
| **Model Fit** | Factor R-squared | 92.32% |
| **Top Risk** | Highest Contributor | NVDA (31.7%) |

---

## Key Takeaways

### 1. Model Selection Matters

| Finding | Implication |
|---------|-------------|
| Historical VaR outperforms Parametric | Fat tails in returns violate normality assumption |
| Historical: 2 violations (GREEN) | Use for regulatory capital |
| Parametric: 6 violations (YELLOW) | Underestimates tail risk |

**Recommendation**: Use **Historical VaR** for regulatory purposes; consider **t-distribution** for Parametric VaR.

### 2. Volatility is Time-Varying

| Finding | Implication |
|---------|-------------|
| GARCH persistence = 0.995 | Volatility shocks last ~6 months |
| ARCH effects significant | Constant volatility models are misspecified |
| Half-life = 145 days | Risk estimates should be updated frequently |

**Recommendation**: Use **GARCH VaR during volatile periods** for more responsive risk management.

### 3. Concentration Risk is Real

| Finding | Implication |
|---------|-------------|
| NVDA: 20% weight, 31.7% risk | Disproportionate risk contribution |
| Tech sector beta = 1.29 | Portfolio amplifies sector moves |
| Tech crash: -38.8% loss | Single sector concentration is dangerous |

**Recommendation**: Consider **diversifying away from high-beta tech stocks** or accept concentrated risk.

### 4. ES Provides Better Tail Information

| Finding | Implication |
|---------|-------------|
| ES is 1.15x VaR | Average tail loss exceeds threshold by 15% |
| FRTB requires 97.5% ES | Industry moving from VaR to ES |
| ES is coherent | Better for portfolio optimization |

**Recommendation**: Report **both VaR and ES** for complete risk picture.

### 5. Stress Testing Reveals Hidden Risks

| Finding | Implication |
|---------|-------------|
| 2022 rate hikes worse than COVID | Recent stress periods are relevant |
| Interest rate insensitivity | Portfolio is pure equity risk |
| Tech crash > Market crash | Sector risk exceeds market risk |

**Recommendation**: Maintain **stressed VaR capital buffer** of 1.8x normal VaR.

### 6. Backtesting Validates Models

| Finding | Implication |
|---------|-------------|
| Historical VaR: p-value = 0.747 | Strong statistical support |
| Violation clustering present | Consider regime-switching models |
| Kupiec + Christoffersen needed | Both coverage and independence matter |

**Recommendation**: Run **both statistical tests** for comprehensive model validation.

---

## Project Structure

```
market_risk_var/
├── README.md                           # This file
├── PROJECT_DETAILS.md                  # Comprehensive technical documentation
├── portfolio_config.json               # Configuration parameters
├── requirements.txt                    # Python dependencies
├── validation_report.py                # Automated validation (40 tests)
│
├── notebooks/                          # Jupyter notebooks (main analysis)
│   ├── 01_Data_Preparation.ipynb       # Data loading and EDA
│   ├── 02_VaR_Methods_Comparison.ipynb # Parametric, Historical, MC VaR
│   ├── 03_Expected_Shortfall.ipynb     # ES/CVaR calculations
│   ├── 04_Factor_Based_VaR.ipynb       # Multi-factor beta-based VaR
│   ├── 05_GARCH_Volatility_VaR.ipynb   # GARCH(1,1) volatility modeling
│   ├── 06_Backtesting_Validation.ipynb # Basel III + statistical tests
│   ├── 07_Stress_Testing.ipynb         # Historical & hypothetical scenarios
│   ├── 08_Component_VaR_Analysis.ipynb # Risk decomposition
│   └── 09_Executive_Summary.ipynb      # Final results dashboard
│
├── src/                                # Reusable Python modules
│   ├── __init__.py
│   ├── data_loader.py                  # Data fetching utilities
│   ├── var_models.py                   # VaR calculation functions
│   ├── backtesting.py                  # Backtesting functions
│   └── visualization.py                # Plotting utilities
│
└── outputs/                            # Generated results
    ├── figures/                        # Saved plots (PNG)
    │   ├── 01_*.png                    # Data preparation visuals
    │   ├── 02_*.png                    # VaR comparison charts
    │   ├── ...                         # Other notebook outputs
    │   └── 09_executive_dashboard.png  # Final dashboard
    ├── prices.csv                      # Historical prices
    ├── asset_returns.csv               # Individual asset returns
    ├── portfolio_returns.csv           # Portfolio returns
    ├── var_results.json                # VaR calculations
    ├── es_results.json                 # Expected Shortfall results
    ├── factor_var_results.json         # Factor model results
    ├── garch_results.json              # GARCH parameters
    ├── backtest_results.json           # Backtesting outcomes
    ├── stress_test_results.json        # Stress test results
    ├── component_var_results.json      # Risk decomposition
    └── executive_summary.json          # Consolidated results
```

---

## Installation

### Prerequisites
- Python 3.8+
- Conda (recommended) or pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/market_risk_var.git
cd market_risk_var
```

2. Create and activate conda environment:
```bash
conda create -n market_risk_var python=3.10
conda activate market_risk_var
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Launch Jupyter:
```bash
jupyter notebook
```

5. Run validation:
```bash
python validation_report.py
```

---

## Methodology

### VaR Formulas

| Method | Formula | Assumptions |
|--------|---------|-------------|
| **Parametric** | VaR = μ + z_α × σ | Normal distribution |
| **Historical** | VaR = Percentile(returns, 1%) | None (empirical) |
| **Monte Carlo** | VaR = Percentile(simulated, 1%) | Specified distribution |
| **GARCH** | VaR = z_α × σ_t (conditional) | Time-varying volatility |

### Expected Shortfall

| Method | Formula |
|--------|---------|
| **Parametric** | ES = μ - σ × φ(z_α) / (1-α) |
| **Historical** | ES = Mean(returns ≤ VaR) |

### GARCH(1,1)

```
σ²_t = ω + α × ε²_{t-1} + β × σ²_{t-1}

Where:
- ω: Long-run variance constant
- α: Reaction to shocks (ARCH effect)
- β: Persistence (GARCH effect)
- Stationarity requires: α + β < 1
```

### Component VaR

```
Marginal VaR_i = z_α × (Σ × w)_i / σ_portfolio
Component VaR_i = w_i × Marginal VaR_i
Sum of Component VaRs = Portfolio VaR (Euler decomposition)
```

---

## Regulatory Framework

### Basel III Traffic Light System

| Zone | Violations (250 days) | Capital Multiplier | Action |
|------|----------------------|-------------------|--------|
| **GREEN** | 0-4 | 3.0 (base) | Model accepted |
| **YELLOW** | 5-9 | 3.4 - 3.85 | Monitoring required |
| **RED** | 10+ | 4.0 | Model rejected |

### FRTB / Basel IV Requirements

| Requirement | Implementation |
|-------------|----------------|
| Expected Shortfall | 97.5% confidence level |
| Holding Period | 10-day for trading book |
| Stress Testing | Historical + hypothetical scenarios |
| Backtesting | 250-day rolling window |

---

## Validation Results

The project includes comprehensive validation with **40 automated tests**:

| Category | Tests | Status |
|----------|-------|--------|
| VaR Formulas | 5 | ✓ All Pass |
| Expected Shortfall | 4 | ✓ All Pass |
| GARCH Model | 4 | ✓ All Pass |
| Component VaR | 4 | ✓ All Pass |
| Factor Model | 3 | ✓ All Pass |
| Backtesting | 5 | ✓ All Pass |
| Stress Testing | 4 | ✓ All Pass |
| Data Integrity | 5 | ✓ All Pass |
| Statistical Properties | 4 | ✓ All Pass |
| Cross-Validation | 2 | ✓ All Pass |

Run `python validation_report.py` to verify all calculations.

---

## Skills Demonstrated

| Category | Skills |
|----------|--------|
| **Risk Management** | VaR, ES, Stress Testing, Backtesting, Risk Decomposition |
| **Statistics** | MLE, Hypothesis Testing, Time Series, Distribution Analysis |
| **Programming** | Python, Pandas, NumPy, SciPy, Statsmodels, Arch |
| **Finance** | Portfolio Theory, Factor Models, Regulatory Compliance |
| **Visualization** | Matplotlib, Seaborn, Dashboard Creation |

---

## References

- Basel Committee on Banking Supervision (BCBS). "Supervisory framework for the use of backtesting." (1996)
- Hull, J.C. "Risk Management and Financial Institutions" (5th Edition)
- Jorion, P. "Value at Risk: The New Benchmark for Managing Financial Risk"
- Christoffersen, P. "Evaluating Interval Forecasts" (1998)
- Kupiec, P. "Techniques for Verifying the Accuracy of Risk Models" (1995)
- Basel III: A global regulatory framework for more resilient banks
- Fundamental Review of the Trading Book (FRTB)

---

## License

MIT License - See LICENSE file for details.

---

## Author

**Hemanth Reddy**
Email: aeddullahemanthreddy1@gmail.com

*Market Risk Analysis Project - December 2025*
