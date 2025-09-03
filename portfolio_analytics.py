"""
Portfolio analytics including correlation analysis, diversification metrics,
and performance attribution.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from data_models import Portfolio, MarketData, CorrelationMatrix, PerformanceAttribution
from risk_metrics import RiskCalculator


class PortfolioAnalytics:
    """Analytics for portfolio correlation, diversification, and attribution."""

    @staticmethod
    def calculate_correlation_matrix(market_data: MarketData) -> CorrelationMatrix:
        """
        Calculate correlation matrix for portfolio holdings.

        Args:
            market_data: Market data with returns

        Returns:
            CorrelationMatrix object
        """
        if market_data.returns.empty:
            return CorrelationMatrix(matrix=pd.DataFrame())

        # Calculate correlation matrix
        corr_matrix = market_data.returns.corr()

        # Calculate diversification score (inverse of average correlation)
        avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        diversification_score = 1 / (1 + abs(avg_corr)) if not np.isnan(avg_corr) else 0.5

        return CorrelationMatrix(
            matrix=corr_matrix,
            diversification_score=diversification_score
        )

    @staticmethod
    def calculate_asset_allocation(portfolio: Portfolio) -> Dict[str, Dict[str, float]]:
        """
        Calculate asset allocation by sector and geography.

        Args:
            portfolio: Portfolio object

        Returns:
            Dictionary with allocation breakdowns
        """
        total_value = portfolio.total_value
        if total_value == 0:
            return {}

        # For now, we'll use a simple allocation by symbol
        # In a real implementation, you'd map symbols to sectors/geographies
        allocation = {
            'by_symbol': portfolio.weights,
            'by_sector': {},  # Would need sector mapping
            'by_geography': {}  # Would need geography mapping
        }

        return allocation

    @staticmethod
    def calculate_diversification_score(portfolio: Portfolio, market_data: MarketData) -> float:
        """
        Calculate portfolio diversification score.

        Args:
            portfolio: Portfolio object
            market_data: Market data

        Returns:
            Diversification score (0-1, higher is better diversified)
        """
        if not portfolio.holdings:
            return 0.0

        weights = list(portfolio.weights.values())

        # Herfindahl-Hirschman Index (HHI)
        hhi = sum(w ** 2 for w in weights)

        # Diversification score = 1 - normalized HHI
        diversification_score = 1 - hhi

        # Adjust based on correlation
        corr_matrix = PortfolioAnalytics.calculate_correlation_matrix(market_data)
        if not corr_matrix.matrix.empty:
            avg_corr = corr_matrix.matrix.values[np.triu_indices_from(corr_matrix.matrix.values, k=1)].mean()
            if not np.isnan(avg_corr):
                # Reduce diversification score if correlations are high
                diversification_score *= (1 - abs(avg_corr))

        return max(0, min(1, diversification_score))

    @staticmethod
    def calculate_performance_attribution(portfolio: Portfolio, market_data: MarketData,
                                        benchmark_symbol: Optional[str] = None) -> PerformanceAttribution:
        """
        Calculate performance attribution by sector, stock, and factors.

        Args:
            portfolio: Portfolio object
            market_data: Market data
            benchmark_symbol: Benchmark symbol for comparison

        Returns:
            PerformanceAttribution object
        """
        if market_data.returns.empty:
            return PerformanceAttribution(total_return=0.0)

        # Calculate portfolio returns
        portfolio_returns = RiskCalculator.calculate_portfolio_returns(portfolio, market_data)

        if portfolio_returns.empty:
            return PerformanceAttribution(total_return=0.0)

        # Calculate total portfolio return
        total_return = (1 + portfolio_returns).prod() - 1

        # Benchmark return
        benchmark_return = None
        excess_return = None
        if benchmark_symbol and market_data.benchmark_returns is not None:
            if benchmark_symbol in market_data.benchmark_returns.columns:
                bench_returns = market_data.benchmark_returns[benchmark_symbol]
                benchmark_return = (1 + bench_returns).prod() - 1
                excess_return = total_return - benchmark_return

        # Stock-level attribution (simplified)
        stock_contributions = {}
        for holding in portfolio.holdings:
            if holding.symbol in market_data.returns.columns:
                stock_returns = market_data.returns[holding.symbol]
                stock_total_return = (1 + stock_returns).prod() - 1
                weight = portfolio.weights.get(holding.symbol, 0)
                stock_contributions[holding.symbol] = stock_total_return * weight

        # Sector attribution (would need sector mapping)
        sector_contributions = {}

        # Factor attribution (simplified - would need factor model)
        factor_contributions = {}

        return PerformanceAttribution(
            total_return=total_return,
            benchmark_return=benchmark_return,
            excess_return=excess_return,
            sector_contributions=sector_contributions,
            stock_contributions=stock_contributions,
            factor_contributions=factor_contributions
        )

    @staticmethod
    def calculate_rolling_performance(portfolio: Portfolio, market_data: MarketData,
                                    window: int = 252) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.

        Args:
            portfolio: Portfolio object
            market_data: Market data
            window: Rolling window size (default 252 trading days)

        Returns:
            DataFrame with rolling metrics
        """
        if market_data.returns.empty:
            return pd.DataFrame()

        portfolio_returns = RiskCalculator.calculate_portfolio_returns(portfolio, market_data)

        if portfolio_returns.empty:
            return pd.DataFrame()

        # Calculate rolling returns
        rolling_returns = portfolio_returns.rolling(window=window).mean() * 252  # Annualized

        # Calculate rolling volatility
        rolling_volatility = portfolio_returns.rolling(window=window).std() * np.sqrt(252)

        # Calculate rolling Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        rolling_sharpe = ((rolling_returns - risk_free_rate) / rolling_volatility).fillna(0)

        # Calculate rolling drawdowns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.rolling(window=window).max()
        rolling_drawdowns = (cumulative_returns - rolling_max) / rolling_max

        result = pd.DataFrame({
            'rolling_return': rolling_returns,
            'rolling_volatility': rolling_volatility,
            'rolling_sharpe': rolling_sharpe,
            'rolling_drawdown': rolling_drawdowns
        })

        return result.dropna()

    @staticmethod
    def identify_outliers(portfolio: Portfolio, market_data: MarketData,
                        threshold: float = 3.0) -> List[str]:
        """
        Identify outlier holdings based on return distribution.

        Args:
            portfolio: Portfolio object
            market_data: Market data
            threshold: Z-score threshold for outliers

        Returns:
            List of outlier symbols
        """
        if market_data.returns.empty:
            return []

        outliers = []

        for holding in portfolio.holdings:
            if holding.symbol in market_data.returns.columns:
                returns = market_data.returns[holding.symbol]
                mean_return = returns.mean()
                std_return = returns.std()

                if std_return > 0:
                    z_score = abs((returns.iloc[-1] - mean_return) / std_return)
                    if z_score > threshold:
                        outliers.append(holding.symbol)

        return outliers

    @staticmethod
    def calculate_risk_contribution(portfolio: Portfolio, market_data: MarketData) -> Dict[str, float]:
        """
        Calculate risk contribution of each holding to portfolio volatility.

        Args:
            portfolio: Portfolio object
            market_data: Market data

        Returns:
            Dictionary of risk contributions by symbol
        """
        if market_data.returns.empty:
            return {}

        symbols = [h.symbol for h in portfolio.holdings]
        weights = np.array([portfolio.weights.get(s, 0) for s in symbols])

        # Get covariance matrix
        cov_matrix = market_data.returns[symbols].cov()

        # Calculate portfolio variance
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix.values, weights))

        if portfolio_variance <= 0:
            return {}

        # Calculate marginal risk contributions
        risk_contributions = {}
        for i, symbol in enumerate(symbols):
            # Marginal contribution to risk
            marginal_risk = np.dot(cov_matrix.values[i], weights) / np.sqrt(portfolio_variance)
            # Total contribution
            total_contribution = weights[i] * marginal_risk
            risk_contributions[symbol] = total_contribution

        return risk_contributions

    @staticmethod
    def cluster_holdings(market_data: MarketData, n_clusters: Optional[int] = None) -> Dict[str, int]:
        """
        Cluster holdings based on return correlation.

        Args:
            market_data: Market data
            n_clusters: Number of clusters (if None, determined automatically)

        Returns:
            Dictionary mapping symbols to cluster IDs
        """
        if market_data.returns.empty or market_data.returns.shape[1] < 2:
            return {}

        # Calculate correlation matrix
        corr_matrix = market_data.returns.corr()

        # Convert to distance matrix
        distance_matrix = 1 - corr_matrix.values

        # Perform hierarchical clustering
        linkage_matrix = linkage(distance_matrix, method='ward')

        if n_clusters is None:
            # Determine optimal number of clusters using elbow method (simplified)
            n_clusters = max(2, int(np.sqrt(len(corr_matrix.columns))))

        # Get cluster assignments
        clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

        cluster_dict = {}
        for i, symbol in enumerate(corr_matrix.columns):
            cluster_dict[symbol] = clusters[i]

        return cluster_dict

    @staticmethod
    def generate_rebalancing_recommendations(portfolio: Portfolio,
                                           target_weights: Dict[str, float],
                                           transaction_cost: float = 0.001) -> Dict[str, float]:
        """
        Generate rebalancing recommendations.

        Args:
            portfolio: Current portfolio
            target_weights: Target portfolio weights
            transaction_cost: Transaction cost as fraction

        Returns:
            Dictionary of recommended trades (positive = buy, negative = sell)
        """
        current_weights = portfolio.weights
        total_value = portfolio.total_value

        if total_value == 0:
            return {}

        recommendations = {}

        for symbol in set(current_weights.keys()) | set(target_weights.keys()):
            current_weight = current_weights.get(symbol, 0)
            target_weight = target_weights.get(symbol, 0)

            weight_difference = target_weight - current_weight
            trade_value = weight_difference * total_value

            # Only recommend if difference is significant
            if abs(trade_value) > transaction_cost * total_value:
                recommendations[symbol] = trade_value

        return recommendations


class VisualizationHelper:
    """Helper class for creating portfolio analytics visualizations."""

    @staticmethod
    def create_correlation_heatmap(corr_matrix: pd.DataFrame, figsize: Tuple[int, int] = (10, 8)):
        """
        Create correlation heatmap visualization.

        Args:
            corr_matrix: Correlation matrix DataFrame
            figsize: Figure size

        Returns:
            Plotly figure object
        """
        if corr_matrix.empty:
            return None

        # Create heatmap using seaborn for matplotlib
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})

        plt.title('Portfolio Correlation Heatmap')
        plt.tight_layout()

        return plt.gcf()

    @staticmethod
    def create_efficient_frontier_plot(expected_returns: np.ndarray,
                                     volatilities: np.ndarray,
                                     sharpe_ratios: np.ndarray,
                                     figsize: Tuple[int, int] = (10, 6)):
        """
        Create efficient frontier plot.

        Args:
            expected_returns: Array of expected returns
            volatilities: Array of volatilities
            sharpe_ratios: Array of Sharpe ratios
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=figsize)

        # Scatter plot of portfolios
        scatter = plt.scatter(volatilities, expected_returns, c=sharpe_ratios,
                            cmap='viridis', alpha=0.6)

        # Find efficient frontier (simplified)
        sorted_indices = np.argsort(volatilities)
        efficient_indices = []
        max_return = -np.inf

        for i in sorted_indices:
            if expected_returns[i] > max_return:
                efficient_indices.append(i)
                max_return = expected_returns[i]

        # Plot efficient frontier
        if efficient_indices:
            eff_vol = volatilities[efficient_indices]
            eff_ret = expected_returns[efficient_indices]
            plt.plot(eff_vol, eff_ret, 'r-', linewidth=2, label='Efficient Frontier')

        plt.colorbar(scatter, label='Sharpe Ratio')
        plt.xlabel('Expected Volatility')
        plt.ylabel('Expected Return')
        plt.title('Efficient Frontier')
        plt.legend()
        plt.grid(True, alpha=0.3)

        return plt.gcf()