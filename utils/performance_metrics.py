"""
Performance Metrics for Portfolio Analysis

This module implements various performance metrics and risk measures
for portfolio evaluation including Sharpe ratio, drawdown, and more.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
import warnings


class PerformanceMetrics:
    """
    Comprehensive performance metrics calculator for portfolios.
    
    This class implements various performance and risk measures including
    return metrics, risk metrics, and risk-adjusted return metrics.
    """
    
    def __init__(self, risk_free_rate: float = 0.02, periods_per_year: int = 252):
        """
        Initialize the Performance Metrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year (252 for daily, 12 for monthly)
        """
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        
    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """
        Calculate returns from price series.
        
        Args:
            prices: Series of asset prices
            
        Returns:
            Series of returns
        """
        return prices.pct_change().dropna()
    
    def calculate_log_returns(self, prices: pd.Series) -> pd.Series:
        """
        Calculate log returns from price series.
        
        Args:
            prices: Series of asset prices
            
        Returns:
            Series of log returns
        """
        return np.log(prices / prices.shift(1)).dropna()
    
    def cumulative_returns(self, returns: pd.Series) -> pd.Series:
        """
        Calculate cumulative returns.
        
        Args:
            returns: Series of returns
            
        Returns:
            Series of cumulative returns
        """
        return (1 + returns).cumprod()
    
    def annualized_return(self, returns: pd.Series) -> float:
        """
        Calculate annualized return.
        
        Args:
            returns: Series of returns
            
        Returns:
            Annualized return
        """
        total_return = (1 + returns).prod() - 1
        n_periods = len(returns)
        return (1 + total_return) ** (self.periods_per_year / n_periods) - 1
    
    def annualized_volatility(self, returns: pd.Series) -> float:
        """
        Calculate annualized volatility.
        
        Args:
            returns: Series of returns
            
        Returns:
            Annualized volatility
        """
        return returns.std() * np.sqrt(self.periods_per_year)
    
    def sharpe_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Series of returns
            
        Returns:
            Sharpe ratio
        """
        excess_returns = returns - self.risk_free_rate / self.periods_per_year
        return np.sqrt(self.periods_per_year) * excess_returns.mean() / returns.std()
    
    def sortino_ratio(self, returns: pd.Series, target_return: float = 0) -> float:
        """
        Calculate Sortino ratio.
        
        Args:
            returns: Series of returns
            target_return: Target return for downside deviation
            
        Returns:
            Sortino ratio
        """
        excess_returns = returns - target_return
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf
        
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
        return np.sqrt(self.periods_per_year) * excess_returns.mean() / downside_deviation
    
    def calmar_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Calmar ratio (annualized return / maximum drawdown).
        
        Args:
            returns: Series of returns
            
        Returns:
            Calmar ratio
        """
        max_dd = self.maximum_drawdown(returns)
        if max_dd == 0:
            return np.inf
        
        annual_return = self.annualized_return(returns)
        return annual_return / abs(max_dd)
    
    def information_ratio(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate information ratio.
        
        Args:
            returns: Series of portfolio returns
            benchmark_returns: Series of benchmark returns
            
        Returns:
            Information ratio
        """
        active_returns = returns - benchmark_returns
        tracking_error = active_returns.std()
        
        if tracking_error == 0:
            return 0
        
        return np.sqrt(self.periods_per_year) * active_returns.mean() / tracking_error
    
    def treynor_ratio(self, returns: pd.Series, market_returns: pd.Series) -> float:
        """
        Calculate Treynor ratio.
        
        Args:
            returns: Series of portfolio returns
            market_returns: Series of market returns
            
        Returns:
            Treynor ratio
        """
        beta = self.beta(returns, market_returns)
        if beta == 0:
            return np.inf
        
        excess_return = returns.mean() - self.risk_free_rate / self.periods_per_year
        return np.sqrt(self.periods_per_year) * excess_return / beta
    
    def beta(self, returns: pd.Series, market_returns: pd.Series) -> float:
        """
        Calculate beta relative to market.
        
        Args:
            returns: Series of portfolio returns
            market_returns: Series of market returns
            
        Returns:
            Beta coefficient
        """
        covariance = np.cov(returns, market_returns)[0, 1]
        market_variance = market_returns.var()
        
        if market_variance == 0:
            return 0
        
        return covariance / market_variance
    
    def alpha(self, returns: pd.Series, market_returns: pd.Series) -> float:
        """
        Calculate Jensen's alpha.
        
        Args:
            returns: Series of portfolio returns
            market_returns: Series of market returns
            
        Returns:
            Alpha (excess return not explained by beta)
        """
        beta_val = self.beta(returns, market_returns)
        expected_return = self.risk_free_rate / self.periods_per_year + beta_val * market_returns.mean()
        actual_return = returns.mean()
        
        return (actual_return - expected_return) * self.periods_per_year
    
    def maximum_drawdown(self, returns: pd.Series) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            returns: Series of returns
            
        Returns:
            Maximum drawdown as a percentage
        """
        cumulative = self.cumulative_returns(returns)
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def drawdown_series(self, returns: pd.Series) -> pd.Series:
        """
        Calculate drawdown series.
        
        Args:
            returns: Series of returns
            
        Returns:
            Series of drawdowns
        """
        cumulative = self.cumulative_returns(returns)
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown
    
    def value_at_risk(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
            
        Returns:
            Value at Risk
        """
        return np.percentile(returns, confidence_level * 100)
    
    def conditional_value_at_risk(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        """
        Calculate Conditional Value at Risk (CVaR).
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level
            
        Returns:
            Conditional Value at Risk
        """
        var = self.value_at_risk(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def skewness(self, returns: pd.Series) -> float:
        """
        Calculate skewness of returns.
        
        Args:
            returns: Series of returns
            
        Returns:
            Skewness coefficient
        """
        return stats.skew(returns)
    
    def kurtosis(self, returns: pd.Series) -> float:
        """
        Calculate kurtosis of returns.
        
        Args:
            returns: Series of returns
            
        Returns:
            Kurtosis coefficient
        """
        return stats.kurtosis(returns)
    
    def jarque_bera_test(self, returns: pd.Series) -> Tuple[float, float]:
        """
        Perform Jarque-Bera test for normality.
        
        Args:
            returns: Series of returns
            
        Returns:
            Tuple of (test_statistic, p_value)
        """
        return stats.jarque_bera(returns)
    
    def win_rate(self, returns: pd.Series) -> float:
        """
        Calculate win rate (percentage of positive returns).
        
        Args:
            returns: Series of returns
            
        Returns:
            Win rate as a percentage
        """
        return (returns > 0).mean() * 100
    
    def profit_factor(self, returns: pd.Series) -> float:
        """
        Calculate profit factor (gross profit / gross loss).
        
        Args:
            returns: Series of returns
            
        Returns:
            Profit factor
        """
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        
        if gross_loss == 0:
            return np.inf
        
        return gross_profit / gross_loss
    
    def recovery_factor(self, returns: pd.Series) -> float:
        """
        Calculate recovery factor (total return / maximum drawdown).
        
        Args:
            returns: Series of returns
            
        Returns:
            Recovery factor
        """
        total_return = (1 + returns).prod() - 1
        max_dd = abs(self.maximum_drawdown(returns))
        
        if max_dd == 0:
            return np.inf
        
        return total_return / max_dd
    
    def calculate_all_metrics(self, returns: pd.Series, 
                            benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Calculate all performance metrics.
        
        Args:
            returns: Series of portfolio returns
            benchmark_returns: Optional series of benchmark returns
            
        Returns:
            Dictionary with all calculated metrics
        """
        metrics = {}
        
        # Basic return metrics
        metrics['total_return'] = (1 + returns).prod() - 1
        metrics['annualized_return'] = self.annualized_return(returns)
        metrics['annualized_volatility'] = self.annualized_volatility(returns)
        
        # Risk-adjusted return metrics
        metrics['sharpe_ratio'] = self.sharpe_ratio(returns)
        metrics['sortino_ratio'] = self.sortino_ratio(returns)
        metrics['calmar_ratio'] = self.calmar_ratio(returns)
        
        # Risk metrics
        metrics['maximum_drawdown'] = self.maximum_drawdown(returns)
        metrics['var_95'] = self.value_at_risk(returns, 0.05)
        metrics['cvar_95'] = self.conditional_value_at_risk(returns, 0.05)
        
        # Distribution metrics
        metrics['skewness'] = self.skewness(returns)
        metrics['kurtosis'] = self.kurtosis(returns)
        
        # Trading metrics
        metrics['win_rate'] = self.win_rate(returns)
        metrics['profit_factor'] = self.profit_factor(returns)
        metrics['recovery_factor'] = self.recovery_factor(returns)
        
        # Benchmark-relative metrics (if benchmark provided)
        if benchmark_returns is not None:
            metrics['information_ratio'] = self.information_ratio(returns, benchmark_returns)
            metrics['beta'] = self.beta(returns, benchmark_returns)
            metrics['alpha'] = self.alpha(returns, benchmark_returns)
            metrics['treynor_ratio'] = self.treynor_ratio(returns, benchmark_returns)
        
        return metrics
    
    def plot_performance_analysis(self, returns: pd.Series, 
                                benchmark_returns: Optional[pd.Series] = None) -> None:
        """
        Plot comprehensive performance analysis.
        
        Args:
            returns: Series of portfolio returns
            benchmark_returns: Optional series of benchmark returns
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Cumulative returns
        cumulative_returns = self.cumulative_returns(returns)
        axes[0, 0].plot(cumulative_returns.index, cumulative_returns.values, 'b-', linewidth=2, label='Portfolio')
        
        if benchmark_returns is not None:
            benchmark_cumulative = self.cumulative_returns(benchmark_returns)
            axes[0, 0].plot(benchmark_cumulative.index, benchmark_cumulative.values, 'r--', linewidth=2, label='Benchmark')
        
        axes[0, 0].set_title('Cumulative Returns')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Drawdown
        drawdown = self.drawdown_series(returns)
        axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        axes[0, 1].plot(drawdown.index, drawdown.values, 'r-', linewidth=1)
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Drawdown')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Return distribution
        axes[0, 2].hist(returns, bins=50, alpha=0.7, density=True, color='blue')
        axes[0, 2].set_title('Return Distribution')
        axes[0, 2].set_xlabel('Returns')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Rolling Sharpe ratio
        rolling_sharpe = returns.rolling(window=252).apply(self.sharpe_ratio)
        axes[1, 0].plot(rolling_sharpe.index, rolling_sharpe.values, 'g-', linewidth=2)
        axes[1, 0].set_title('Rolling Sharpe Ratio (252-day)')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Sharpe Ratio')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Rolling volatility
        rolling_vol = returns.rolling(window=252).std() * np.sqrt(self.periods_per_year)
        axes[1, 1].plot(rolling_vol.index, rolling_vol.values, 'orange', linewidth=2)
        axes[1, 1].set_title('Rolling Volatility (252-day)')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Annualized Volatility')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Rolling beta (if benchmark provided)
        if benchmark_returns is not None:
            rolling_beta = returns.rolling(window=252).apply(lambda x: self.beta(x, benchmark_returns.loc[x.index]))
            axes[1, 2].plot(rolling_beta.index, rolling_beta.values, 'purple', linewidth=2)
            axes[1, 2].set_title('Rolling Beta (252-day)')
            axes[1, 2].set_xlabel('Date')
            axes[1, 2].set_ylabel('Beta')
            axes[1, 2].grid(True, alpha=0.3)
        else:
            # Alternative: Rolling maximum drawdown
            rolling_max_dd = returns.rolling(window=252).apply(self.maximum_drawdown)
            axes[1, 2].plot(rolling_max_dd.index, rolling_max_dd.values, 'purple', linewidth=2)
            axes[1, 2].set_title('Rolling Max Drawdown (252-day)')
            axes[1, 2].set_xlabel('Date')
            axes[1, 2].set_ylabel('Maximum Drawdown')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, returns: pd.Series, 
                       benchmark_returns: Optional[pd.Series] = None,
                       portfolio_name: str = "Portfolio") -> str:
        """
        Generate a comprehensive performance report.
        
        Args:
            returns: Series of portfolio returns
            benchmark_returns: Optional series of benchmark returns
            portfolio_name: Name of the portfolio
            
        Returns:
            Formatted performance report
        """
        metrics = self.calculate_all_metrics(returns, benchmark_returns)
        
        report = f"""
{'='*60}
{portfolio_name} Performance Report
{'='*60}

RETURN METRICS:
{'-'*30}
Total Return: {metrics['total_return']:.2%}
Annualized Return: {metrics['annualized_return']:.2%}
Annualized Volatility: {metrics['annualized_volatility']:.2%}

RISK-ADJUSTED RETURNS:
{'-'*30}
Sharpe Ratio: {metrics['sharpe_ratio']:.3f}
Sortino Ratio: {metrics['sortino_ratio']:.3f}
Calmar Ratio: {metrics['calmar_ratio']:.3f}

RISK METRICS:
{'-'*30}
Maximum Drawdown: {metrics['maximum_drawdown']:.2%}
Value at Risk (95%): {metrics['var_95']:.2%}
Conditional VaR (95%): {metrics['cvar_95']:.2%}

DISTRIBUTION METRICS:
{'-'*30}
Skewness: {metrics['skewness']:.3f}
Kurtosis: {metrics['kurtosis']:.3f}
Win Rate: {metrics['win_rate']:.1f}%

TRADING METRICS:
{'-'*30}
Profit Factor: {metrics['profit_factor']:.3f}
Recovery Factor: {metrics['recovery_factor']:.3f}
"""
        
        if benchmark_returns is not None:
            report += f"""
BENCHMARK-RELATIVE METRICS:
{'-'*30}
Information Ratio: {metrics['information_ratio']:.3f}
Beta: {metrics['beta']:.3f}
Alpha: {metrics['alpha']:.2%}
Treynor Ratio: {metrics['treynor_ratio']:.3f}
"""
        
        report += f"{'='*60}"
        
        return report


def calculate_portfolio_metrics(returns: pd.Series, 
                              benchmark_returns: Optional[pd.Series] = None,
                              risk_free_rate: float = 0.02) -> Dict[str, float]:
    """
    Convenience function to calculate portfolio metrics.
    
    Args:
        returns: Series of portfolio returns
        benchmark_returns: Optional series of benchmark returns
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Dictionary with all calculated metrics
    """
    calculator = PerformanceMetrics(risk_free_rate=risk_free_rate)
    return calculator.calculate_all_metrics(returns, benchmark_returns)


if __name__ == "__main__":
    # Example usage
    print("Performance Metrics Module")
    print("Use PerformanceMetrics class to calculate comprehensive portfolio performance measures")
