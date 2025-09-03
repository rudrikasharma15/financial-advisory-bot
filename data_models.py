"""
Data models for portfolio analytics and risk management.
Defines core data structures for portfolios, holdings, and market data.
"""

from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from datetime import datetime


@dataclass
class Holding:
    """Represents a single holding in a portfolio."""
    symbol: str
    quantity: float
    purchase_price: float
    purchase_date: datetime
    current_price: Optional[float] = None
    sector: Optional[str] = None
    market_value: Optional[float] = None

    def __post_init__(self):
        if self.market_value is None and self.current_price is not None:
            self.market_value = self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> Optional[float]:
        """Calculate unrealized profit/loss."""
        if self.current_price is None:
            return None
        return (self.current_price - self.purchase_price) * self.quantity

    @property
    def unrealized_pnl_percent(self) -> Optional[float]:
        """Calculate unrealized profit/loss percentage."""
        if self.purchase_price == 0 or self.current_price is None:
            return None
        return ((self.current_price - self.purchase_price) / self.purchase_price) * 100


@dataclass
class Portfolio:
    """Represents a complete investment portfolio."""
    name: str
    holdings: List[Holding] = field(default_factory=list)
    cash: float = 0.0
    created_date: datetime = field(default_factory=datetime.now)
    benchmark_symbol: Optional[str] = None

    @property
    def total_value(self) -> float:
        """Calculate total portfolio value including cash."""
        holdings_value = sum(h.market_value or 0 for h in self.holdings)
        return holdings_value + self.cash

    @property
    def total_invested(self) -> float:
        """Calculate total amount invested (excluding cash)."""
        return sum(h.quantity * h.purchase_price for h in self.holdings)

    @property
    def total_unrealized_pnl(self) -> float:
        """Calculate total unrealized profit/loss."""
        return sum(h.unrealized_pnl or 0 for h in self.holdings)

    @property
    def weights(self) -> Dict[str, float]:
        """Calculate portfolio weights by symbol."""
        total_value = self.total_value
        if total_value == 0:
            return {}
        return {h.symbol: (h.market_value or 0) / total_value for h in self.holdings}

    def add_holding(self, holding: Holding):
        """Add a holding to the portfolio."""
        self.holdings.append(holding)

    def remove_holding(self, symbol: str):
        """Remove a holding from the portfolio."""
        self.holdings = [h for h in self.holdings if h.symbol != symbol]

    def update_prices(self, price_data: Dict[str, float]):
        """Update current prices for holdings."""
        for holding in self.holdings:
            if holding.symbol in price_data:
                holding.current_price = price_data[holding.symbol]
                holding.market_value = holding.quantity * holding.current_price


@dataclass
class MarketData:
    """Container for market data used in analytics."""
    prices: pd.DataFrame  # Historical prices for all symbols
    returns: pd.DataFrame  # Historical returns
    benchmark_prices: Optional[pd.DataFrame] = None
    benchmark_returns: Optional[pd.DataFrame] = None

    def __post_init__(self):
        if not self.returns.empty and self.prices.empty:
            # Calculate prices from returns if only returns provided
            self.prices = (1 + self.returns).cumprod()
        elif not self.prices.empty and self.returns.empty:
            # Calculate returns from prices
            self.returns = self.prices.pct_change().dropna()


@dataclass
class RiskMetrics:
    """Container for various risk metrics."""
    var_95: Optional[float] = None  # Value at Risk (95% confidence)
    cvar_95: Optional[float] = None  # Conditional VaR (95% confidence)
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    beta: Optional[float] = None
    volatility: Optional[float] = None
    alpha: Optional[float] = None


@dataclass
class PerformanceAttribution:
    """Performance attribution analysis results."""
    total_return: float
    benchmark_return: Optional[float] = None
    excess_return: Optional[float] = None
    sector_contributions: Dict[str, float] = field(default_factory=dict)
    stock_contributions: Dict[str, float] = field(default_factory=dict)
    factor_contributions: Dict[str, float] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Results from portfolio optimization."""
    optimal_weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    efficient_frontier: Optional[List[Dict[str, float]]] = None


@dataclass
class CorrelationMatrix:
    """Correlation analysis results."""
    matrix: pd.DataFrame
    heatmap_data: Optional[Dict] = None  # For visualization
    diversification_score: Optional[float] = None


@dataclass
class RebalancingRecommendation:
    """Portfolio rebalancing recommendations."""
    current_weights: Dict[str, float]
    target_weights: Dict[str, float]
    trades_required: Dict[str, float]  # Positive = buy, negative = sell
    estimated_cost: float
    reason: str