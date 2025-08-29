"""
Portfolio Optimizer using CVXPY for Constrained Allocation

This module implements portfolio optimization algorithms using CVXPY
to solve constrained allocation problems with various objective functions.
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Dict, List, Tuple, Optional, Union
import warnings


class PortfolioOptimizer:
    """
    Portfolio optimizer using CVXPY for various optimization objectives.
    
    This class implements different portfolio optimization strategies including
    mean-variance optimization, risk parity, and maximum Sharpe ratio.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize the Portfolio Optimizer.
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculations
        """
        self.risk_free_rate = risk_free_rate
        self.weights = None
        self.optimization_result = None
        
    def mean_variance_optimization(self, 
                                 expected_returns: np.ndarray,
                                 covariance_matrix: np.ndarray,
                                 target_return: Optional[float] = None,
                                 risk_aversion: float = 1.0,
                                 constraints: Optional[List] = None) -> Dict:
        """
        Mean-variance optimization.
        
        Args:
            expected_returns: Array of expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            target_return: Target portfolio return (if None, maximize Sharpe ratio)
            risk_aversion: Risk aversion parameter
            constraints: Additional constraints as list of CVXPY constraints
            
        Returns:
            Dictionary with optimization results
        """
        n_assets = len(expected_returns)
        
        # Decision variables
        weights = cp.Variable(n_assets)
        
        # Objective function
        if target_return is not None:
            # Minimize risk subject to return constraint
            objective = cp.Minimize(cp.quad_form(weights, covariance_matrix))
            constraints_list = [cp.sum(weights) == 1, 
                              expected_returns @ weights >= target_return]
        else:
            # Maximize utility (return - risk_aversion * risk)
            portfolio_return = expected_returns @ weights
            portfolio_risk = cp.quad_form(weights, covariance_matrix)
            objective = cp.Maximize(portfolio_return - 0.5 * risk_aversion * portfolio_risk)
            constraints_list = [cp.sum(weights) == 1]
        
        # Add additional constraints
        if constraints:
            constraints_list.extend(constraints)
        
        # Solve the problem
        problem = cp.Problem(objective, constraints_list)
        problem.solve()
        
        if problem.status not in ["infeasible", "unbounded"]:
            self.weights = weights.value
            self.optimization_result = {
                'status': problem.status,
                'weights': weights.value,
                'portfolio_return': expected_returns @ weights.value,
                'portfolio_risk': np.sqrt(weights.value.T @ covariance_matrix @ weights.value),
                'sharpe_ratio': (expected_returns @ weights.value - self.risk_free_rate) / 
                               np.sqrt(weights.value.T @ covariance_matrix @ weights.value)
            }
        else:
            raise ValueError(f"Optimization failed with status: {problem.status}")
        
        return self.optimization_result
    
    def risk_parity_optimization(self, 
                               covariance_matrix: np.ndarray,
                               target_risk: Optional[float] = None,
                               constraints: Optional[List] = None) -> Dict:
        """
        Risk parity optimization (equal risk contribution).
        
        Args:
            covariance_matrix: Covariance matrix of returns
            target_risk: Target portfolio risk (if None, minimize total risk)
            constraints: Additional constraints
            
        Returns:
            Dictionary with optimization results
        """
        n_assets = len(covariance_matrix)
        
        # Decision variables
        weights = cp.Variable(n_assets)
        
        # Risk contributions
        portfolio_risk = cp.sqrt(cp.quad_form(weights, covariance_matrix))
        risk_contributions = []
        
        for i in range(n_assets):
            # Risk contribution of asset i
            rc_i = weights[i] * (covariance_matrix[i, :] @ weights) / portfolio_risk
            risk_contributions.append(rc_i)
        
        # Objective: minimize variance of risk contributions
        risk_contrib_array = cp.hstack(risk_contributions)
        objective = cp.Minimize(cp.sum_squares(risk_contrib_array - cp.mean(risk_contrib_array)))
        
        # Constraints
        constraints_list = [cp.sum(weights) == 1, weights >= 0]
        
        if target_risk is not None:
            constraints_list.append(portfolio_risk <= target_risk)
        
        # Add additional constraints
        if constraints:
            constraints_list.extend(constraints)
        
        # Solve the problem
        problem = cp.Problem(objective, constraints_list)
        problem.solve()
        
        if problem.status not in ["infeasible", "unbounded"]:
            self.weights = weights.value
            self.optimization_result = {
                'status': problem.status,
                'weights': weights.value,
                'portfolio_risk': np.sqrt(weights.value.T @ covariance_matrix @ weights.value),
                'risk_contributions': [rc.value for rc in risk_contributions]
            }
        else:
            raise ValueError(f"Optimization failed with status: {problem.status}")
        
        return self.optimization_result
    
    def maximum_sharpe_optimization(self, 
                                  expected_returns: np.ndarray,
                                  covariance_matrix: np.ndarray,
                                  constraints: Optional[List] = None) -> Dict:
        """
        Maximum Sharpe ratio optimization.
        
        Args:
            expected_returns: Array of expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            constraints: Additional constraints
            
        Returns:
            Dictionary with optimization results
        """
        n_assets = len(expected_returns)
        
        # Decision variables
        weights = cp.Variable(n_assets)
        
        # Portfolio return and risk
        portfolio_return = expected_returns @ weights
        portfolio_risk = cp.sqrt(cp.quad_form(weights, covariance_matrix))
        
        # Sharpe ratio (maximize)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
        
        # Objective: maximize Sharpe ratio
        objective = cp.Maximize(sharpe_ratio)
        
        # Constraints
        constraints_list = [cp.sum(weights) == 1, weights >= 0]
        
        # Add additional constraints
        if constraints:
            constraints_list.extend(constraints)
        
        # Solve the problem
        problem = cp.Problem(objective, constraints_list)
        problem.solve()
        
        if problem.status not in ["infeasible", "unbounded"]:
            self.weights = weights.value
            self.optimization_result = {
                'status': problem.status,
                'weights': weights.value,
                'portfolio_return': expected_returns @ weights.value,
                'portfolio_risk': np.sqrt(weights.value.T @ covariance_matrix @ weights.value),
                'sharpe_ratio': (expected_returns @ weights.value - self.risk_free_rate) / 
                               np.sqrt(weights.value.T @ covariance_matrix @ weights.value)
            }
        else:
            raise ValueError(f"Optimization failed with status: {problem.status}")
        
        return self.optimization_result
    
    def black_litterman_optimization(self, 
                                   market_caps: np.ndarray,
                                   covariance_matrix: np.ndarray,
                                   risk_aversion: float = 2.5,
                                   views: Optional[Dict] = None,
                                   view_confidences: Optional[np.ndarray] = None,
                                   tau: float = 0.05) -> Dict:
        """
        Black-Litterman optimization with market equilibrium and views.
        
        Args:
            market_caps: Market capitalizations (weights)
            covariance_matrix: Covariance matrix of returns
            risk_aversion: Risk aversion parameter
            views: Dictionary of views (e.g., {'AAPL': 0.05, 'GOOGL': -0.03})
            view_confidences: Confidence in views (inverse of variance)
            tau: Scaling parameter for prior uncertainty
            
        Returns:
            Dictionary with optimization results
        """
        n_assets = len(market_caps)
        
        # Market equilibrium returns
        market_weights = market_caps / np.sum(market_caps)
        pi = risk_aversion * covariance_matrix @ market_weights
        
        # Prior covariance
        prior_cov = tau * covariance_matrix
        
        if views is not None:
            # Create view matrix and vector
            view_assets = list(views.keys())
            view_values = list(views.values())
            
            P = np.zeros((len(views), n_assets))
            Q = np.array(view_values)
            
            for i, asset in enumerate(view_assets):
                # Find asset index (assuming assets are in order)
                asset_idx = i  # This should be properly mapped
                P[i, asset_idx] = 1
            
            # View uncertainty matrix
            if view_confidences is None:
                view_confidences = np.ones(len(views)) * 0.1
            
            Omega = np.diag(1 / view_confidences)
            
            # Black-Litterman formula
            M1 = np.linalg.inv(prior_cov)
            M2 = P.T @ np.linalg.inv(Omega) @ P
            M3 = np.linalg.inv(prior_cov) @ pi
            M4 = P.T @ np.linalg.inv(Omega) @ Q
            
            mu_bl = np.linalg.inv(M1 + M2) @ (M3 + M4)
            sigma_bl = np.linalg.inv(M1 + M2)
            
            # Use Black-Litterman estimates
            expected_returns = mu_bl
            covariance_matrix = sigma_bl
        else:
            expected_returns = pi
            covariance_matrix = prior_cov
        
        # Now optimize using mean-variance
        return self.mean_variance_optimization(expected_returns, covariance_matrix)
    
    def add_constraints(self, 
                       constraint_type: str, 
                       **kwargs) -> cp.Constraint:
        """
        Create common portfolio constraints.
        
        Args:
            constraint_type: Type of constraint
            **kwargs: Constraint parameters
            
        Returns:
            CVXPY constraint
        """
        if constraint_type == "long_only":
            return cp.Variable(len(kwargs.get('n_assets', 1))) >= 0
        
        elif constraint_type == "weight_bounds":
            n_assets = kwargs.get('n_assets')
            min_weight = kwargs.get('min_weight', 0)
            max_weight = kwargs.get('max_weight', 1)
            weights = cp.Variable(n_assets)
            return [weights >= min_weight, weights <= max_weight]
        
        elif constraint_type == "sector_constraint":
            sector_weights = kwargs.get('sector_weights')
            max_sector_weight = kwargs.get('max_sector_weight', 0.3)
            return cp.sum(sector_weights) <= max_sector_weight
        
        elif constraint_type == "turnover_constraint":
            current_weights = kwargs.get('current_weights')
            max_turnover = kwargs.get('max_turnover', 0.1)
            weights = cp.Variable(len(current_weights))
            return cp.sum(cp.abs(weights - current_weights)) <= max_turnover
        
        else:
            raise ValueError(f"Unknown constraint type: {constraint_type}")
    
    def get_efficient_frontier(self, 
                             expected_returns: np.ndarray,
                             covariance_matrix: np.ndarray,
                             return_range: Tuple[float, float],
                             n_points: int = 50,
                             constraints: Optional[List] = None) -> Dict:
        """
        Generate efficient frontier.
        
        Args:
            expected_returns: Array of expected returns
            covariance_matrix: Covariance matrix
            return_range: Tuple of (min_return, max_return)
            n_points: Number of points on frontier
            constraints: Additional constraints
            
        Returns:
            Dictionary with efficient frontier data
        """
        min_return, max_return = return_range
        target_returns = np.linspace(min_return, max_return, n_points)
        
        frontier_data = []
        
        for target_return in target_returns:
            try:
                result = self.mean_variance_optimization(
                    expected_returns, 
                    covariance_matrix, 
                    target_return=target_return,
                    constraints=constraints
                )
                frontier_data.append({
                    'return': result['portfolio_return'],
                    'risk': result['portfolio_risk'],
                    'sharpe': result['sharpe_ratio']
                })
            except:
                continue
        
        return {
            'returns': [d['return'] for d in frontier_data],
            'risks': [d['risk'] for d in frontier_data],
            'sharpe_ratios': [d['sharpe'] for d in frontier_data]
        }
    
    def get_weights_summary(self) -> pd.DataFrame:
        """
        Get summary of current portfolio weights.
        
        Returns:
            DataFrame with weight summary
        """
        if self.weights is None:
            raise ValueError("No optimization has been performed yet")
        
        summary = pd.DataFrame({
            'Asset': [f'Asset_{i}' for i in range(len(self.weights))],
            'Weight': self.weights,
            'Weight_Pct': self.weights * 100
        })
        
        return summary.sort_values('Weight', ascending=False)


def optimize_portfolio(returns: pd.DataFrame,
                     method: str = "mean_variance",
                     **kwargs) -> PortfolioOptimizer:
    """
    Convenience function to optimize portfolio.
    
    Args:
        returns: DataFrame of asset returns
        method: Optimization method
        **kwargs: Additional arguments for the optimizer
        
    Returns:
        Optimized PortfolioOptimizer instance
    """
    optimizer = PortfolioOptimizer()
    
    # Calculate expected returns and covariance
    expected_returns = returns.mean().values
    covariance_matrix = returns.cov().values
    
    if method == "mean_variance":
        optimizer.mean_variance_optimization(expected_returns, covariance_matrix, **kwargs)
    elif method == "risk_parity":
        optimizer.risk_parity_optimization(covariance_matrix, **kwargs)
    elif method == "max_sharpe":
        optimizer.maximum_sharpe_optimization(expected_returns, covariance_matrix, **kwargs)
    elif method == "black_litterman":
        optimizer.black_litterman_optimization(
            kwargs.get('market_caps', np.ones(len(expected_returns))),
            covariance_matrix,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown optimization method: {method}")
    
    return optimizer


if __name__ == "__main__":
    # Example usage
    print("Portfolio Optimization Module")
    print("Use PortfolioOptimizer class to solve constrained allocation problems using CVXPY")
