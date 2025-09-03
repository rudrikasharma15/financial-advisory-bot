"""
Risk metrics calculations for portfolio analysis.
Implements VaR, CVaR, Sharpe ratio, Sortino ratio, Calmar ratio, Beta, and other risk measures.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple, Union
import warnings

from data_models import Portfolio, MarketData, RiskMetrics


class RiskCalculator:
    """Calculates various risk metrics for portfolios."""

    @staticmethod
    def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
        """Calculate returns from price series."""
        if method == 'log':
            return np.log(prices / prices.shift(1)).dropna()
        else:
            return prices.pct_change().dropna()

    @staticmethod
    def calculate_portfolio_returns(portfolio: Portfolio, market_data: MarketData) -> pd.Series:
        """Calculate portfolio returns from holdings and market data."""
        if market_data.returns.empty:
            return pd.Series(dtype=float)

        # Get symbols in portfolio
        symbols = [h.symbol for h in portfolio.holdings]
        weights = portfolio.weights

        if not symbols or not weights:
            return pd.Series(dtype=float)

        # Filter returns to only include portfolio symbols
        available_symbols = [s for s in symbols if s in market_data.returns.columns]
        if not available_symbols:
            return pd.Series(dtype=float)

        portfolio_returns = market_data.returns[available_symbols].copy()

        # Calculate weighted portfolio returns
        portfolio_return_series = pd.Series(0.0, index=portfolio_returns.index)

        for symbol in available_symbols:
            if symbol in weights:
                portfolio_return_series += portfolio_returns[symbol] * weights[symbol]

        return portfolio_return_series

    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.95,
                     method: str = 'historical') -> float:
        """
        Calculate Value at Risk (VaR).

        Args:
            returns: Series of portfolio returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            method: 'historical', 'parametric', or 'monte_carlo'

        Returns:
            VaR value (positive number representing potential loss)
        """
        if returns.empty or len(returns) < 30:
            return 0.0

        try:
            if method == 'parametric':
                # Parametric VaR using normal distribution
                mean_return = returns.mean()
                std_return = returns.std()
                if std_return == 0:
                    return 0.0
                z_score = stats.norm.ppf(1 - confidence_level)
                var = -(mean_return + z_score * std_return)
            elif method == 'monte_carlo':
                # Monte Carlo VaR (simplified)
                n_simulations = 10000
                simulated_returns = np.random.normal(returns.mean(), returns.std(), n_simulations)
                var = -np.percentile(simulated_returns, (1 - confidence_level) * 100)
            else:
                # Historical VaR
                var = -np.percentile(returns, (1 - confidence_level) * 100)

            return float(max(0, var))  # Ensure non-negative and return as float
        except:
            return 0.0

    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence_level: float = 0.95,
                      method: str = 'historical') -> float:
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall).

        Args:
            returns: Series of portfolio returns
            confidence_level: Confidence level
            method: 'historical' or 'parametric'

        Returns:
            CVaR value (positive number representing expected loss beyond VaR)
        """
        if returns.empty or len(returns) < 30:
            return 0.0

        try:
            if method == 'parametric':
                # Parametric CVaR
                mean_return = returns.mean()
                std_return = returns.std()
                if std_return == 0:
                    return 0.0
                z_score = stats.norm.ppf(1 - confidence_level)
                cvar = -(mean_return + (std_return * stats.norm.pdf(z_score) / (1 - confidence_level)))
            else:
                # Historical CVaR
                var_threshold = np.percentile(returns, (1 - confidence_level) * 100)
                tail_losses = returns[returns <= var_threshold]
                if len(tail_losses) > 0:
                    cvar = -tail_losses.mean()
                else:
                    cvar = -var_threshold

            return float(max(0, cvar))
        except:
            return 0.0

    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02,
                              annualize: bool = True) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns: Series of portfolio returns
            risk_free_rate: Risk-free rate (annual)
            annualize: Whether to annualize the ratio

        Returns:
            Sharpe ratio
        """
        if returns.empty or len(returns) < 30:
            return 0.0

        excess_returns = returns - risk_free_rate / (252 if annualize else 1)
        mean_excess_return = excess_returns.mean()
        std_excess_return = excess_returns.std()

        if std_excess_return == 0:
            return 0.0

        sharpe = mean_excess_return / std_excess_return

        if annualize:
            sharpe *= np.sqrt(252)  # Annualize assuming daily returns

        return sharpe

    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02,
                               annualize: bool = True) -> float:
        """
        Calculate Sortino ratio (downside deviation instead of total volatility).

        Args:
            returns: Series of portfolio returns
            risk_free_rate: Risk-free rate (annual)
            annualize: Whether to annualize the ratio

        Returns:
            Sortino ratio
        """
        if returns.empty or len(returns) < 30:
            return 0.0

        excess_returns = returns - risk_free_rate / (252 if annualize else 1)
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0

        downside_deviation = np.sqrt(np.mean(downside_returns ** 2))

        if downside_deviation == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0

        sortino = excess_returns.mean() / downside_deviation

        if annualize:
            sortino *= np.sqrt(252)

        return sortino

    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Calmar ratio (annual return / maximum drawdown).

        Args:
            returns: Series of portfolio returns
            risk_free_rate: Risk-free rate (annual)

        Returns:
            Calmar ratio
        """
        if returns.empty or len(returns) < 30:
            return 0.0

        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()

        # Calculate drawdowns
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdowns.min()

        if max_drawdown >= 0:
            return 0.0

        # Calculate annualized return
        total_return = cumulative_returns.iloc[-1] - 1
        years = len(returns) / 252  # Assuming daily returns
        annualized_return = (1 + total_return) ** (1 / years) - 1

        calmar = annualized_return / abs(max_drawdown)
        return calmar

    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> float:
        """
        Calculate maximum drawdown.

        Args:
            returns: Series of portfolio returns

        Returns:
            Maximum drawdown (positive number)
        """
        if returns.empty:
            return 0.0

        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(drawdowns.min())

        return max_drawdown

    @staticmethod
    def calculate_beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate portfolio beta relative to benchmark.

        Args:
            portfolio_returns: Portfolio returns series
            benchmark_returns: Benchmark returns series

        Returns:
            Beta coefficient
        """
        if portfolio_returns.empty or benchmark_returns.empty:
            return 1.0

        # Align the series
        common_index = portfolio_returns.index.intersection(benchmark_returns.index)
        if len(common_index) < 30:
            return 1.0

        port_returns = portfolio_returns.loc[common_index]
        bench_returns = benchmark_returns.loc[common_index]

        # Calculate covariance and variance
        covariance = np.cov(port_returns, bench_returns)[0, 1]
        variance = np.var(bench_returns)

        if variance == 0:
            return 1.0

        beta = covariance / variance
        return beta

    @staticmethod
    def calculate_alpha(portfolio_returns: pd.Series, benchmark_returns: pd.Series,
                       risk_free_rate: float = 0.02) -> float:
        """
        Calculate portfolio alpha.

        Args:
            portfolio_returns: Portfolio returns series
            benchmark_returns: Benchmark returns series
            risk_free_rate: Risk-free rate

        Returns:
            Alpha (excess return not explained by beta)
        """
        if portfolio_returns.empty or benchmark_returns.empty:
            return 0.0

        beta = RiskCalculator.calculate_beta(portfolio_returns, benchmark_returns)

        port_excess_return = portfolio_returns.mean() - risk_free_rate / 252
        bench_excess_return = benchmark_returns.mean() - risk_free_rate / 252

        alpha = port_excess_return - beta * bench_excess_return
        return alpha * 252  # Annualize

    @staticmethod
    def calculate_volatility(returns: pd.Series, annualize: bool = True) -> float:
        """
        Calculate portfolio volatility (standard deviation).

        Args:
            returns: Series of portfolio returns
            annualize: Whether to annualize the volatility

        Returns:
            Volatility (standard deviation)
        """
        if returns.empty:
            return 0.0

        volatility = returns.std()

        if annualize:
            volatility *= np.sqrt(252)  # Assuming daily returns

        return volatility

    @classmethod
    def calculate_all_risk_metrics(cls, portfolio: Portfolio, market_data: MarketData,
                                  risk_free_rate: float = 0.02) -> RiskMetrics:
        """
        Calculate all risk metrics for a portfolio.

        Args:
            portfolio: Portfolio object
            market_data: Market data container
            risk_free_rate: Risk-free rate

        Returns:
            RiskMetrics object with all calculated metrics
        """
        try:
            # Calculate portfolio returns
            portfolio_returns = cls.calculate_portfolio_returns(portfolio, market_data)

            if portfolio_returns.empty or len(portfolio_returns) < 30:
                return RiskMetrics(
                    var_95=0.0,
                    cvar_95=0.0,
                    sharpe_ratio=0.0,
                    sortino_ratio=0.0,
                    calmar_ratio=0.0,
                    max_drawdown=0.0,
                    volatility=0.0
                )

            # Calculate individual metrics
            var_95 = cls.calculate_var(portfolio_returns, 0.95)
            cvar_95 = cls.calculate_cvar(portfolio_returns, 0.95)
            sharpe_ratio = cls.calculate_sharpe_ratio(portfolio_returns, risk_free_rate)
            sortino_ratio = cls.calculate_sortino_ratio(portfolio_returns, risk_free_rate)
            calmar_ratio = cls.calculate_calmar_ratio(portfolio_returns, risk_free_rate)
            max_drawdown = cls.calculate_max_drawdown(portfolio_returns)
            volatility = cls.calculate_volatility(portfolio_returns)

            # Calculate beta and alpha if benchmark data available
            beta = None
            alpha = None
            if market_data.benchmark_returns is not None and not market_data.benchmark_returns.empty:
                beta = cls.calculate_beta(portfolio_returns, market_data.benchmark_returns.iloc[:, 0])
                alpha = cls.calculate_alpha(portfolio_returns, market_data.benchmark_returns.iloc[:, 0], risk_free_rate)

            return RiskMetrics(
                var_95=var_95,
                cvar_95=cvar_95,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                max_drawdown=max_drawdown,
                beta=beta,
                volatility=volatility,
                alpha=alpha
            )

        except Exception as e:
            warnings.warn(f"Error calculating risk metrics: {str(e)}")
            return RiskMetrics(
                var_95=0.0,
                cvar_95=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                max_drawdown=0.0,
                volatility=0.0
            )


def calculate_tracking_error(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Calculate tracking error (standard deviation of excess returns).

    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns

    Returns:
        Tracking error
    """
    if portfolio_returns.empty or benchmark_returns.empty:
        return 0.0

    common_index = portfolio_returns.index.intersection(benchmark_returns.index)
    if len(common_index) < 30:
        return 0.0

    excess_returns = portfolio_returns.loc[common_index] - benchmark_returns.loc[common_index]
    tracking_error = excess_returns.std() * np.sqrt(252)  # Annualize

    return tracking_error


def calculate_information_ratio(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Calculate information ratio (excess return / tracking error).

    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns

    Returns:
        Information ratio
    """
    tracking_error = calculate_tracking_error(portfolio_returns, benchmark_returns)

    if tracking_error == 0:
        return 0.0

    # Calculate annualized excess return
    common_index = portfolio_returns.index.intersection(benchmark_returns.index)
    if len(common_index) < 30:
        return 0.0

    excess_returns = portfolio_returns.loc[common_index] - benchmark_returns.loc[common_index]
    annualized_excess_return = excess_returns.mean() * 252

    return annualized_excess_return / tracking_error