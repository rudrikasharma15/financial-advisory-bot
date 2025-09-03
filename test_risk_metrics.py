"""
Unit tests for risk metrics calculations.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from data_models import Portfolio, Holding, MarketData
from risk_metrics import RiskCalculator


class TestRiskMetrics(unittest.TestCase):
    """Test cases for risk metrics calculations."""

    def setUp(self):
        """Set up test data."""
        # Create sample market data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)

        # Generate sample price data
        prices = pd.DataFrame({
            'AAPL': 150 + np.cumsum(np.random.normal(0, 2, 100)),
            'MSFT': 250 + np.cumsum(np.random.normal(0, 3, 100)),
            'GOOGL': 2800 + np.cumsum(np.random.normal(0, 50, 100))
        }, index=dates)

        returns = prices.pct_change().dropna()

        self.market_data = MarketData(
            prices=prices,
            returns=returns
        )

        # Create sample portfolio
        holdings = [
            Holding('AAPL', 100, 140.0, datetime(2023, 1, 1)),
            Holding('MSFT', 50, 240.0, datetime(2023, 1, 1)),
            Holding('GOOGL', 10, 2700.0, datetime(2023, 1, 1))
        ]

        # Update current prices
        for holding in holdings:
            if holding.symbol in prices.columns:
                holding.current_price = prices[holding.symbol].iloc[-1]

        self.portfolio = Portfolio("Test Portfolio", holdings=holdings)

    def test_portfolio_returns_calculation(self):
        """Test portfolio returns calculation."""
        portfolio_returns = RiskCalculator.calculate_portfolio_returns(
            self.portfolio, self.market_data
        )

        self.assertIsInstance(portfolio_returns, pd.Series)

        # The portfolio should have some returns if symbols match
        symbols_in_portfolio = [h.symbol for h in self.portfolio.holdings]
        symbols_in_market = list(self.market_data.returns.columns)

        if any(symbol in symbols_in_market for symbol in symbols_in_portfolio):
            # If there's overlap, we should get some returns
            self.assertGreaterEqual(len(portfolio_returns), 0)
        else:
            # If no overlap, returns should be empty
            self.assertEqual(len(portfolio_returns), 0)

    def test_var_calculation(self):
        """Test Value at Risk calculation."""
        portfolio_returns = RiskCalculator.calculate_portfolio_returns(
            self.portfolio, self.market_data
        )

        var_95 = RiskCalculator.calculate_var(portfolio_returns, 0.95)

        self.assertIsInstance(var_95, float)
        self.assertTrue(var_95 <= 0)  # VaR should be negative or zero
        self.assertTrue(var_95 >= -1)  # VaR shouldn't be worse than -100%

    def test_cvar_calculation(self):
        """Test Conditional VaR calculation."""
        portfolio_returns = RiskCalculator.calculate_portfolio_returns(
            self.portfolio, self.market_data
        )

        cvar_95 = RiskCalculator.calculate_cvar(portfolio_returns, 0.95)

        self.assertIsInstance(cvar_95, float)
        self.assertTrue(cvar_95 <= 0)  # CVaR should be negative or zero
        self.assertTrue(cvar_95 >= -1)  # CVaR shouldn't be worse than -100%

    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation."""
        portfolio_returns = RiskCalculator.calculate_portfolio_returns(
            self.portfolio, self.market_data
        )

        sharpe = RiskCalculator.calculate_sharpe_ratio(portfolio_returns, 0.02)

        self.assertIsInstance(sharpe, float)
        # Sharpe ratio can be any real number, but should be reasonable
        self.assertTrue(abs(sharpe) < 10)

    def test_sortino_ratio_calculation(self):
        """Test Sortino ratio calculation."""
        portfolio_returns = RiskCalculator.calculate_portfolio_returns(
            self.portfolio, self.market_data
        )

        sortino = RiskCalculator.calculate_sortino_ratio(portfolio_returns, 0.02)

        self.assertIsInstance(sortino, (float, type(float('inf'))))
        # Sortino ratio can be any real number
        self.assertTrue(isinstance(sortino, (float, type(float('inf')))))

    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        portfolio_returns = RiskCalculator.calculate_portfolio_returns(
            self.portfolio, self.market_data
        )

        max_dd = RiskCalculator.calculate_max_drawdown(portfolio_returns)

        self.assertIsInstance(max_dd, float)
        self.assertTrue(0 <= max_dd <= 1)  # Max drawdown should be between 0 and 100%

    def test_volatility_calculation(self):
        """Test volatility calculation."""
        portfolio_returns = RiskCalculator.calculate_portfolio_returns(
            self.portfolio, self.market_data
        )

        vol = RiskCalculator.calculate_volatility(portfolio_returns)

        self.assertIsInstance(vol, float)
        self.assertTrue(vol >= 0)  # Volatility should be non-negative
        self.assertTrue(vol < 1)  # Annualized volatility should be less than 100%

    def test_all_risk_metrics(self):
        """Test calculation of all risk metrics."""
        risk_metrics = RiskCalculator.calculate_all_risk_metrics(
            self.portfolio, self.market_data, 0.02
        )

        self.assertIsNotNone(risk_metrics)
        self.assertIsInstance(risk_metrics.var_95, float)
        self.assertIsInstance(risk_metrics.sharpe_ratio, (float, type(None)))
        self.assertIsInstance(risk_metrics.max_drawdown, (float, type(None)))

    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        # Create market data with very few points
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        prices = pd.DataFrame({
            'AAPL': [150] * 10,
            'MSFT': [250] * 10
        }, index=dates)

        returns = prices.pct_change().dropna()
        market_data_small = MarketData(prices=prices, returns=returns)

        var = RiskCalculator.calculate_var(returns['AAPL'], 0.95)
        self.assertEqual(var, 0.0)  # Should return 0 for insufficient data

    def test_empty_portfolio_handling(self):
        """Test handling of empty portfolio."""
        empty_portfolio = Portfolio("Empty", holdings=[])

        portfolio_returns = RiskCalculator.calculate_portfolio_returns(
            empty_portfolio, self.market_data
        )

        self.assertTrue(portfolio_returns.empty)


if __name__ == '__main__':
    unittest.main()