"""
Portfolio optimization algorithms using convex optimization.
Implements Markowitz optimization, risk parity, and other portfolio optimization techniques.
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Dict, List, Optional, Tuple, Union
import warnings

from data_models import Portfolio, MarketData, OptimizationResult
from risk_metrics import RiskCalculator


class PortfolioOptimizer:
    """Portfolio optimization using convex optimization techniques."""

    def __init__(self, market_data: MarketData, risk_free_rate: float = 0.02):
        """
        Initialize optimizer with market data.

        Args:
            market_data: Market data with returns
            risk_free_rate: Risk-free rate for optimization
        """
        self.market_data = market_data
        self.risk_free_rate = risk_free_rate
        self.expected_returns = None
        self.covariance_matrix = None

        if not market_data.returns.empty:
            self.expected_returns = market_data.returns.mean() * 252  # Annualize
            self.covariance_matrix = market_data.returns.cov() * 252  # Annualize

    def optimize_mean_variance(self, target_return: Optional[float] = None,
                             target_volatility: Optional[float] = None,
                             max_weight: float = 1.0, min_weight: float = 0.0) -> OptimizationResult:
        """
        Mean-variance optimization (Markowitz optimization).

        Args:
            target_return: Target portfolio return (if None, maximize Sharpe)
            target_volatility: Target portfolio volatility
            max_weight: Maximum weight per asset
            min_weight: Minimum weight per asset

        Returns:
            OptimizationResult with optimal weights
        """
        if self.expected_returns is None or self.covariance_matrix is None:
            return OptimizationResult(
                optimal_weights={},
                expected_return=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0
            )

        n_assets = len(self.expected_returns)
        symbols = self.expected_returns.index.tolist()

        # Variables
        weights = cp.Variable(n_assets)

        # Constraints
        constraints = [
            cp.sum(weights) == 1,  # Fully invested
            weights >= min_weight,  # Minimum weight
            weights <= max_weight   # Maximum weight
        ]

        if target_return is not None:
            # Minimize volatility for target return
            portfolio_return = self.expected_returns.values @ weights
            portfolio_volatility = cp.quad_form(weights, self.covariance_matrix.values)
            objective = cp.Minimize(portfolio_volatility)
            constraints.append(portfolio_return >= target_return)
        elif target_volatility is not None:
            # Maximize return for target volatility
            portfolio_return = self.expected_returns.values @ weights
            portfolio_volatility = cp.quad_form(weights, self.covariance_matrix.values)
            objective = cp.Maximize(portfolio_return)
            constraints.append(portfolio_volatility <= target_volatility**2)
        else:
            # Maximize Sharpe ratio (simplified)
            portfolio_return = self.expected_returns.values @ weights
            portfolio_volatility = cp.quad_form(weights, self.covariance_matrix.values)
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / cp.sqrt(portfolio_volatility)
            objective = cp.Maximize(sharpe_ratio)

        # Solve optimization problem
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.ECOS)

            if problem.status != cp.OPTIMAL:
                warnings.warn(f"Optimization failed with status: {problem.status}")
                return OptimizationResult(
                    optimal_weights={},
                    expected_return=0.0,
                    expected_volatility=0.0,
                    sharpe_ratio=0.0
                )

            optimal_weights = weights.value
            expected_return = float(self.expected_returns.values @ optimal_weights)
            expected_volatility = float(np.sqrt(optimal_weights @ self.covariance_matrix.values @ optimal_weights))
            sharpe_ratio = (expected_return - self.risk_free_rate) / expected_volatility

            weights_dict = {symbols[i]: optimal_weights[i] for i in range(n_assets)}

            return OptimizationResult(
                optimal_weights=weights_dict,
                expected_return=expected_return,
                expected_volatility=expected_volatility,
                sharpe_ratio=sharpe_ratio
            )

        except Exception as e:
            warnings.warn(f"Optimization error: {str(e)}")
            return OptimizationResult(
                optimal_weights={},
                expected_return=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0
            )

    def optimize_risk_parity(self, max_weight: float = 1.0, min_weight: float = 0.0) -> OptimizationResult:
        """
        Risk parity optimization - equal risk contribution from each asset.

        Args:
            max_weight: Maximum weight per asset
            min_weight: Minimum weight per asset

        Returns:
            OptimizationResult with optimal weights
        """
        if self.covariance_matrix is None:
            return OptimizationResult(
                optimal_weights={},
                expected_return=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0
            )

        n_assets = len(self.covariance_matrix)
        symbols = self.covariance_matrix.index.tolist()

        # Variables
        weights = cp.Variable(n_assets)

        # Risk contributions (marginal risk * weight)
        portfolio_volatility = cp.quad_form(weights, self.covariance_matrix.values)
        marginal_risks = self.covariance_matrix.values @ weights
        risk_contributions = cp.multiply(marginal_risks, weights)

        # Target risk contribution (equal for all assets)
        target_risk = portfolio_volatility / n_assets

        # Constraints
        constraints = [
            cp.sum(weights) == 1,
            weights >= min_weight,
            weights <= max_weight
        ]

        # Risk parity constraints (all risk contributions equal)
        for i in range(1, n_assets):
            constraints.append(risk_contributions[i-1] == risk_contributions[i])

        # Objective: minimize portfolio volatility (subject to risk parity)
        objective = cp.Minimize(portfolio_volatility)

        # Solve
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.ECOS)

            if problem.status != cp.OPTIMAL:
                warnings.warn(f"Risk parity optimization failed with status: {problem.status}")
                return OptimizationResult(
                    optimal_weights={},
                    expected_return=0.0,
                    expected_volatility=0.0,
                    sharpe_ratio=0.0
                )

            optimal_weights = weights.value
            expected_return = float(self.expected_returns.values @ optimal_weights)
            expected_volatility = float(np.sqrt(optimal_weights @ self.covariance_matrix.values @ optimal_weights))
            sharpe_ratio = (expected_return - self.risk_free_rate) / expected_volatility

            weights_dict = {symbols[i]: optimal_weights[i] for i in range(n_assets)}

            return OptimizationResult(
                optimal_weights=weights_dict,
                expected_return=expected_return,
                expected_volatility=expected_volatility,
                sharpe_ratio=sharpe_ratio
            )

        except Exception as e:
            warnings.warn(f"Risk parity optimization error: {str(e)}")
            return OptimizationResult(
                optimal_weights={},
                expected_return=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0
            )

    def optimize_minimum_variance(self, max_weight: float = 1.0, min_weight: float = 0.0) -> OptimizationResult:
        """
        Minimum variance portfolio optimization.

        Args:
            max_weight: Maximum weight per asset
            min_weight: Minimum weight per asset

        Returns:
            OptimizationResult with optimal weights
        """
        return self.optimize_mean_variance(target_return=None, target_volatility=None,
                                        max_weight=max_weight, min_weight=min_weight)

    def calculate_efficient_frontier(self, n_points: int = 50,
                                   max_weight: float = 1.0, min_weight: float = 0.0) -> List[OptimizationResult]:
        """
        Calculate efficient frontier with multiple portfolios.

        Args:
            n_points: Number of points on the frontier
            max_weight: Maximum weight per asset
            min_weight: Minimum weight per asset

        Returns:
            List of OptimizationResult objects representing the efficient frontier
        """
        if self.expected_returns is None or self.covariance_matrix is None:
            return []

        # Find minimum and maximum possible returns
        min_return_result = self.optimize_minimum_variance(max_weight, min_weight)
        max_return_result = self.optimize_mean_variance(target_volatility=0.5, max_weight=max_weight, min_weight=min_weight)

        if not min_return_result.optimal_weights or not max_return_result.optimal_weights:
            return []

        min_return = min_return_result.expected_return
        max_return = max_return_result.expected_return

        if min_return >= max_return:
            return [min_return_result]

        # Generate target returns between min and max
        target_returns = np.linspace(min_return, max_return, n_points)

        efficient_frontier = []
        for target_return in target_returns:
            result = self.optimize_mean_variance(target_return=target_return,
                                              max_weight=max_weight, min_weight=min_weight)
            if result.optimal_weights:
                efficient_frontier.append(result)

        return efficient_frontier

    def optimize_max_sharpe(self, max_weight: float = 1.0, min_weight: float = 0.0) -> OptimizationResult:
        """
        Maximize Sharpe ratio portfolio.

        Args:
            max_weight: Maximum weight per asset
            min_weight: Minimum weight per asset

        Returns:
            OptimizationResult with optimal weights
        """
        return self.optimize_mean_variance(target_return=None, target_volatility=None,
                                        max_weight=max_weight, min_weight=min_weight)

    def rebalance_portfolio(self, current_portfolio: Portfolio,
                          target_weights: Dict[str, float],
                          transaction_cost: float = 0.001) -> Dict[str, float]:
        """
        Calculate trades needed to rebalance portfolio to target weights.

        Args:
            current_portfolio: Current portfolio
            target_weights: Target weights dictionary
            transaction_cost: Transaction cost as fraction

        Returns:
            Dictionary of trades (positive = buy, negative = sell)
        """
        current_weights = current_portfolio.weights
        total_value = current_portfolio.total_value

        if total_value == 0:
            return {}

        trades = {}
        total_buy = 0
        total_sell = 0

        # Calculate required trades
        for symbol in set(current_weights.keys()) | set(target_weights.keys()):
            current_weight = current_weights.get(symbol, 0)
            target_weight = target_weights.get(symbol, 0)

            weight_diff = target_weight - current_weight
            trade_value = weight_diff * total_value

            if abs(trade_value) > transaction_cost * total_value:
                trades[symbol] = trade_value
                if trade_value > 0:
                    total_buy += trade_value
                else:
                    total_sell += abs(trade_value)

        # Adjust for cash flow (assuming we can use cash to fund buys)
        cash_needed = total_buy - total_sell
        if cash_needed > 0 and current_portfolio.cash < cash_needed:
            # Scale down buys proportionally
            scale_factor = current_portfolio.cash / cash_needed
            for symbol, trade_value in trades.items():
                if trade_value > 0:
                    trades[symbol] = trade_value * scale_factor

        return trades

    def black_litterman_adjustment(self, views: Dict[str, float],
                                 confidence_levels: Dict[str, float],
                                 tau: float = 0.05) -> np.ndarray:
        """
        Apply Black-Litterman model to adjust expected returns based on views.

        Args:
            views: Dictionary of expected returns for specific assets
            confidence_levels: Confidence levels for each view (0-1)
            tau: Uncertainty parameter

        Returns:
            Adjusted expected returns array
        """
        if self.expected_returns is None or self.covariance_matrix is None:
            return np.array([])

        n_assets = len(self.expected_returns)
        symbols = self.expected_returns.index.tolist()

        # Prior expected returns and covariance
        pi = self.expected_returns.values
        sigma = self.covariance_matrix.values

        # Create pick matrix P and view vector Q
        view_symbols = list(views.keys())
        n_views = len(view_symbols)

        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)

        for i, symbol in enumerate(view_symbols):
            if symbol in symbols:
                symbol_idx = symbols.index(symbol)
                P[i, symbol_idx] = 1.0
                Q[i] = views[symbol]

        # Confidence matrix Omega
        omega = np.zeros((n_views, n_views))
        for i, symbol in enumerate(view_symbols):
            if symbol in symbols:
                symbol_idx = symbols.index(symbol)
                omega[i, i] = confidence_levels.get(symbol, 0.5) * sigma[symbol_idx, symbol_idx]

        # Black-Litterman formula
        try:
            sigma_inv = np.linalg.inv(sigma)
            omega_inv = np.linalg.inv(omega)

            # Posterior expected returns
            temp1 = sigma_inv + tau * P.T @ omega_inv @ P
            temp2 = np.linalg.inv(temp1)
            temp3 = sigma_inv @ pi + tau * P.T @ omega_inv @ Q

            adjusted_returns = temp2 @ temp3

            return adjusted_returns

        except np.linalg.LinAlgError:
            warnings.warn("Black-Litterman adjustment failed due to matrix inversion error")
            return pi


def calculate_portfolio_metrics(weights: np.ndarray, expected_returns: np.ndarray,
                              covariance_matrix: np.ndarray, risk_free_rate: float = 0.02) -> Dict[str, float]:
    """
    Calculate portfolio metrics for given weights.

    Args:
        weights: Portfolio weights array
        expected_returns: Expected returns array
        covariance_matrix: Covariance matrix
        risk_free_rate: Risk-free rate

    Returns:
        Dictionary of portfolio metrics
    """
    portfolio_return = weights @ expected_returns
    portfolio_volatility = np.sqrt(weights @ covariance_matrix @ weights)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

    return {
        'expected_return': portfolio_return,
        'volatility': portfolio_volatility,
        'sharpe_ratio': sharpe_ratio
    }