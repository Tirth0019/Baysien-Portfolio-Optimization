"""
Hidden Markov Model for Regime Detection in Financial Returns

This module implements a Hidden Markov Model to identify different market regimes
in financial time series data and classify observations into these regimes.
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, Dict, Any


class HMMRegimeDetector:
    """
    Hidden Markov Model for detecting market regimes in financial returns.
    
    This class implements a Gaussian HMM to identify different market states
    such as bull markets, bear markets, and sideways markets.
    """
    
    def __init__(self, n_regimes: int = 3, random_state: int = 42):
        """
        Initialize the HMM Regime Detector.
        
        Args:
            n_regimes: Number of hidden states/regimes to detect
            random_state: Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, returns: pd.Series) -> 'HMMRegimeDetector':
        """
        Fit the HMM model to the return data.
        
        Args:
            returns: Pandas Series of financial returns
            
        Returns:
            Self for method chaining
        """
        # Reshape data for HMM (n_samples, n_features)
        X = returns.values.reshape(-1, 1)
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize and fit the HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            random_state=self.random_state,
            covariance_type="full",
            n_iter=1000
        )
        
        self.model.fit(X_scaled)
        self.is_fitted = True
        
        return self
    
    def predict_regimes(self, returns: pd.Series) -> np.ndarray:
        """
        Predict regime labels for the given returns.
        
        Args:
            returns: Pandas Series of financial returns
            
        Returns:
            Array of regime labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        X = returns.values.reshape(-1, 1)
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)
    
    def get_regime_probabilities(self, returns: pd.Series) -> np.ndarray:
        """
        Get probability of each regime for each observation.
        
        Args:
            returns: Pandas Series of financial returns
            
        Returns:
            Array of shape (n_samples, n_regimes) with regime probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting probabilities")
            
        X = returns.values.reshape(-1, 1)
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict_proba(X_scaled)
    
    def get_regime_characteristics(self) -> Dict[str, Any]:
        """
        Get characteristics of each detected regime.
        
        Returns:
            Dictionary containing regime means, variances, and transition matrix
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting characteristics")
            
        # Get regime means and variances (in original scale)
        means = self.scaler.inverse_transform(self.model.means_)
        covariances = self.model.covars_ * (self.scaler.scale_ ** 2)
        
        return {
            'means': means.flatten(),
            'variances': covariances.flatten(),
            'transition_matrix': self.model.transmat_,
            'stationary_distribution': self.model.get_stationary_distribution()
        }
    
    def plot_regime_analysis(self, returns: pd.Series, regime_labels: np.ndarray) -> None:
        """
        Plot the regime analysis results.
        
        Args:
            returns: Pandas Series of financial returns
            regime_labels: Array of regime labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting")
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Returns with regime coloring
        for regime in range(self.n_regimes):
            mask = regime_labels == regime
            axes[0, 0].scatter(
                returns.index[mask], 
                returns[mask], 
                alpha=0.6, 
                label=f'Regime {regime}',
                s=20
            )
        axes[0, 0].set_title('Returns by Regime')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Returns')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Regime transition heatmap
        transition_matrix = self.model.transmat_
        sns.heatmap(
            transition_matrix, 
            annot=True, 
            cmap='Blues', 
            ax=axes[0, 1],
            xticklabels=[f'R{i}' for i in range(self.n_regimes)],
            yticklabels=[f'R{i}' for i in range(self.n_regimes)]
        )
        axes[0, 1].set_title('Regime Transition Matrix')
        
        # Plot 3: Regime distribution
        regime_counts = pd.Series(regime_labels).value_counts().sort_index()
        axes[1, 0].bar(range(self.n_regimes), regime_counts.values)
        axes[1, 0].set_title('Regime Distribution')
        axes[1, 0].set_xlabel('Regime')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_xticks(range(self.n_regimes))
        
        # Plot 4: Regime characteristics
        characteristics = self.get_regime_characteristics()
        means = characteristics['means']
        variances = characteristics['variances']
        
        x_pos = np.arange(self.n_regimes)
        axes[1, 1].bar(x_pos - 0.2, means, 0.4, label='Mean', alpha=0.7)
        axes[1, 1].bar(x_pos + 0.2, np.sqrt(variances), 0.4, label='Std Dev', alpha=0.7)
        axes[1, 1].set_title('Regime Characteristics')
        axes[1, 1].set_xlabel('Regime')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath: str) -> None:
        """
        Save the fitted HMM model to disk.
        
        Args:
            filepath: Path where to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
            
        import joblib
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'n_regimes': self.n_regimes,
            'random_state': self.random_state
        }
        joblib.dump(model_data, filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'HMMRegimeDetector':
        """
        Load a fitted HMM model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded HMMRegimeDetector instance
        """
        import joblib
        model_data = joblib.load(filepath)
        
        instance = cls(
            n_regimes=model_data['n_regimes'],
            random_state=model_data['random_state']
        )
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.is_fitted = True
        
        return instance


def detect_regimes_from_returns(returns: pd.Series, n_regimes: int = 3) -> Tuple[np.ndarray, HMMRegimeDetector]:
    """
    Convenience function to detect regimes from returns.
    
    Args:
        returns: Pandas Series of financial returns
        n_regimes: Number of regimes to detect
        
    Returns:
        Tuple of (regime_labels, fitted_model)
    """
    detector = HMMRegimeDetector(n_regimes=n_regimes)
    detector.fit(returns)
    regime_labels = detector.predict_regimes(returns)
    
    return regime_labels, detector


if __name__ == "__main__":
    # Example usage
    print("HMM Regime Detection Module")
    print("Use HMMRegimeDetector class to detect market regimes in financial returns")
