# Market Risk VaR Analysis Package
# This package contains reusable modules for VaR calculations, backtesting, and visualization

from .data_loader import (
    get_price_data,
    compute_log_returns,
    compute_simple_returns,
    compute_portfolio_returns,
    load_config,
    get_summary_statistics
)

from .var_models import (
    parametric_var,
    historical_var,
    monte_carlo_var,
    parametric_es,
    historical_es,
    monte_carlo_es,
    scale_var,
    marginal_var,
    component_var
)

from .backtesting import (
    basel_zone,
    backtest_var,
    kupiec_test,
    christoffersen_test,
    rolling_var_backtest
)

from .visualization import (
    plot_var_distribution,
    plot_backtest,
    plot_rolling_var,
    plot_component_var,
    plot_stress_scenarios,
    plot_correlation_matrix
)
