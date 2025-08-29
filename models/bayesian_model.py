"""
Bayesian Models for Financial Return Estimation using PyMC

This module implements Bayesian models to estimate return distributions
and uncertainty in financial time series data.
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, Optional, List
from scipy import stats


class BayesianReturnEstimator:
    """
    Bayesian estimator for financial returns using PyMC.
    
    This class implements various Bayesian models to estimate return distributions
    including normal, student-t, and regime-switching models.
    """
    
    def __init__(self, model_type: str = "normal"):
        """
        Initialize the Bayesian Return Estimator.
        
        Args:
            model_type: Type of model ("normal", "student_t", "regime_switching")
        """
        self.model_type = model_type
        self.model = None
        self.trace = None
        self.is_fitted = False
        
    def fit_normal_model(self, returns: pd.Series) -> pm.Model:
        """
        Fit a normal distribution model to returns.
        
        Args:
            returns: Pandas Series of financial returns
            
        Returns:
            PyMC model
        """
        with pm.Model() as model:
            # Priors
            mu = pm.Normal('mu', mu=0, sigma=0.1)
            sigma = pm.HalfNormal('sigma', sigma=0.1)
            
            # Likelihood
            likelihood = pm.Normal('returns', mu=mu, sigma=sigma, observed=returns.values)
            
        return model
    
    def fit_student_t_model(self, returns: pd.Series) -> pm.Model:
        """
        Fit a student-t distribution model to returns (handles fat tails).
        
        Args:
            returns: Pandas Series of financial returns
            
        Returns:
            PyMC model
        """
        with pm.Model() as model:
            # Priors
            mu = pm.Normal('mu', mu=0, sigma=0.1)
            sigma = pm.HalfNormal('sigma', sigma=0.1)
            nu = pm.Gamma('nu', alpha=2, beta=0.1)
            
            # Likelihood
            likelihood = pm.StudentT('returns', nu=nu, mu=mu, sigma=sigma, observed=returns.values)
            
        return model
    
    def fit_regime_switching_model(self, returns: pd.Series, n_regimes: int = 2) -> pm.Model:
        """
        Fit a regime-switching model to returns.
        
        Args:
            returns: Pandas Series of financial returns
            n_regimes: Number of regimes
            
        Returns:
            PyMC model
        """
        with pm.Model() as model:
            # Regime transition probabilities
            p_transition = pm.Dirichlet('p_transition', a=np.ones(n_regimes), shape=(n_regimes, n_regimes))
            
            # Regime-specific parameters
            mu_regimes = pm.Normal('mu_regimes', mu=0, sigma=0.1, shape=n_regimes)
            sigma_regimes = pm.HalfNormal('sigma_regimes', sigma=0.1, shape=n_regimes)
            
            # Hidden regime states
            regime_states = pm.Categorical('regime_states', p=pm.math.softmax(pm.Normal('regime_probs', 0, 1, shape=n_regimes)), shape=len(returns))
            
            # Likelihood
            likelihood = pm.Normal('returns', 
                                mu=mu_regimes[regime_states], 
                                sigma=sigma_regimes[regime_states], 
                                observed=returns.values)
            
        return model
    
    def fit(self, returns: pd.Series, **kwargs) -> 'BayesianReturnEstimator':
        """
        Fit the Bayesian model to the return data.
        
        Args:
            returns: Pandas Series of financial returns
            **kwargs: Additional arguments for specific model types
            
        Returns:
            Self for method chaining
        """
        if self.model_type == "normal":
            self.model = self.fit_normal_model(returns)
        elif self.model_type == "student_t":
            self.model = self.fit_student_t_model(returns)
        elif self.model_type == "regime_switching":
            n_regimes = kwargs.get('n_regimes', 2)
            self.model = self.fit_regime_switching_model(returns, n_regimes)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Sample from posterior
        with self.model:
            self.trace = pm.sample(
                draws=2000,
                tune=1000,
                chains=4,
                return_inferencedata=True,
                random_seed=42
            )
        
        self.is_fitted = True
        return self
    
    def get_posterior_summary(self) -> pd.DataFrame:
        """
        Get summary statistics of the posterior distributions.
        
        Returns:
            DataFrame with posterior summary statistics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting posterior summary")
            
        return az.summary(self.trace)
    
    def predict_returns(self, n_samples: int = 1000) -> np.ndarray:
        """
        Generate predictions from the fitted model.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Array of predicted returns
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        with self.model:
            ppc = pm.sample_posterior_predictive(self.trace, samples=n_samples, return_inferencedata=True)
            
        return ppc.posterior_predictive['returns'].values
    
    def get_return_distribution(self, n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the estimated return distribution.
        
        Args:
            n_points: Number of points for the distribution
            
        Returns:
            Tuple of (x_values, probability_density)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting distribution")
            
        if self.model_type == "normal":
            mu = self.trace.posterior['mu'].mean().values
            sigma = self.trace.posterior['sigma'].mean().values
            
            x = np.linspace(mu - 4*sigma, mu + 4*sigma, n_points)
            pdf = stats.norm.pdf(x, mu, sigma)
            
        elif self.model_type == "student_t":
            mu = self.trace.posterior['mu'].mean().values
            sigma = self.trace.posterior['sigma'].mean().values
            nu = self.trace.posterior['nu'].mean().values
            
            x = np.linspace(mu - 4*sigma, mu + 4*sigma, n_points)
            pdf = stats.t.pdf(x, nu, mu, sigma)
            
        else:
            # For regime switching, use mixture of normals
            mu_regimes = self.trace.posterior['mu_regimes'].mean().values
            sigma_regimes = self.trace.posterior['sigma_regimes'].mean().values
            
            x = np.linspace(mu_regimes.min() - 4*sigma_regimes.max(), 
                           mu_regimes.max() + 4*sigma_regimes.max(), n_points)
            pdf = np.zeros_like(x)
            
            for i in range(len(mu_regimes)):
                pdf += stats.norm.pdf(x, mu_regimes[i], sigma_regimes[i]) / len(mu_regimes)
        
        return x, pdf
    
    def plot_posterior_analysis(self) -> None:
        """
        Plot posterior analysis including trace plots and posterior distributions.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting")
            
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Trace plot
        az.plot_trace(self.trace, axes=axes[0, :])
        axes[0, 0].set_title('Trace Plot')
        
        # Plot 2: Posterior distributions
        az.plot_posterior(self.trace, axes=axes[1, 0])
        axes[1, 0].set_title('Posterior Distributions')
        
        # Plot 3: Return distribution
        x, pdf = self.get_return_distribution()
        axes[1, 1].plot(x, pdf, 'b-', linewidth=2)
        axes[1, 1].set_title('Estimated Return Distribution')
        axes[1, 1].set_xlabel('Returns')
        axes[1, 1].set_ylabel('Probability Density')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_credible_intervals(self, confidence: float = 0.95) -> Dict[str, Any]:
        """
        Get credible intervals for model parameters.
        
        Args:
            confidence: Confidence level (e.g., 0.95 for 95% CI)
            
        Returns:
            Dictionary with credible intervals for each parameter
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting credible intervals")
            
        summary = az.summary(self.trace, credible_interval=confidence)
        
        intervals = {}
        for var in summary.index:
            lower = summary.loc[var, f'hdi_{int(confidence*100)}%']
            upper = summary.loc[var, f'hdi_{int(confidence*100)}%']
            intervals[var] = {'lower': lower, 'upper': upper}
        
        return intervals
    
    def save_model(self, filepath: str) -> None:
        """
        Save the fitted Bayesian model to disk.
        
        Args:
            filepath: Path where to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
            
        model_data = {
            'model_type': self.model_type,
            'trace': self.trace,
            'model': self.model
        }
        az.to_netcdf(self.trace, filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'BayesianReturnEstimator':
        """
        Load a fitted Bayesian model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded BayesianReturnEstimator instance
        """
        trace = az.from_netcdf(filepath)
        
        instance = cls()
        instance.trace = trace
        instance.is_fitted = True
        
        return instance


def estimate_returns_bayesian(returns: pd.Series, model_type: str = "normal", **kwargs) -> BayesianReturnEstimator:
    """
    Convenience function to estimate returns using Bayesian methods.
    
    Args:
        returns: Pandas Series of financial returns
        model_type: Type of Bayesian model to use
        **kwargs: Additional arguments for the model
        
    Returns:
        Fitted BayesianReturnEstimator instance
    """
    estimator = BayesianReturnEstimator(model_type=model_type)
    estimator.fit(returns, **kwargs)
    
    return estimator


if __name__ == "__main__":
    # Example usage
    print("Bayesian Return Estimation Module")
    print("Use BayesianReturnEstimator class to estimate return distributions using PyMC")
