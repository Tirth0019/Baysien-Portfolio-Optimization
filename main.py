"""
Main Runner Script for Bayesian Portfolio Optimization

This script orchestrates the complete portfolio optimization pipeline:
1. Load data → 2. Detect regime → 3. Estimate returns → 4. Optimize → 5. Evaluate
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our custom modules
from models.hmm_model import HMMRegimeDetector
from models.bayesian_model import BayesianReturnEstimator
from models.optimizer import PortfolioOptimizer
from utils.performance_metrics import PerformanceMetrics
from utils.data_loader import DataLoader


class BayesianPortfolioOptimizer:
    """
    Main class that orchestrates the complete portfolio optimization pipeline.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize the portfolio optimizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.setup_logging()
        
        # Initialize components
        self.data_loader = DataLoader()
        self.regime_detector = None
        self.bayesian_estimator = None
        self.optimizer = None
        self.performance_calculator = None
        
        # Results storage
        self.portfolio_data = None
        self.returns_data = None
        self.regime_labels = None
        self.optimization_results = None
        self.performance_metrics = None
        
    def _get_default_config(self) -> dict:
        """Get default configuration parameters."""
        return {
            'data_source': 'yahoo',  # 'yahoo', 'csv', 'api'
            'tickers': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'],
            'start_date': (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d'),
            'end_date': datetime.now().strftime('%Y-%m-%d'),
            'n_regimes': 3,
            'bayesian_model_type': 'student_t',
            'optimization_method': 'max_sharpe',
            'risk_aversion': 1.0,
            'risk_free_rate': 0.02,
            'rebalance_frequency': 'monthly',
            'transaction_costs': 0.001,
            'constraints': {
                'long_only': True,
                'max_weight': 0.3,
                'min_weight': 0.0
            }
        }
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('portfolio_optimization.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_data(self) -> bool:
        """
        Load historical price data for the specified assets.
        
        Returns:
            True if data loaded successfully, False otherwise
        """
        try:
            self.logger.info("Loading historical price data...")
            
            if self.config['data_source'] == 'yahoo':
                self.portfolio_data = self.data_loader.load_yahoo_data(
                    self.config['tickers'],
                    self.config['start_date'],
                    self.config['end_date']
                )
            elif self.config['data_source'] == 'csv':
                self.portfolio_data = self.data_loader.load_csv_data(
                    os.path.join('data', 'SP500.csv')
                )
            else:
                raise ValueError(f"Unsupported data source: {self.config['data_source']}")
            
            if self.portfolio_data is not None and not self.portfolio_data.empty:
                self.returns_data = self.portfolio_data.pct_change().dropna()
                self.logger.info(f"Data loaded successfully: {len(self.returns_data)} observations for {len(self.returns_data.columns)} assets")
                return True
            else:
                self.logger.error("Failed to load data")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            return False
    
    def detect_regimes(self) -> bool:
        """
        Detect market regimes using Hidden Markov Models.
        
        Returns:
            True if regime detection successful, False otherwise
        """
        try:
            self.logger.info("Detecting market regimes...")
            
            # Use portfolio-level returns for regime detection
            portfolio_returns = self.returns_data.mean(axis=1)
            
            self.regime_detector = HMMRegimeDetector(
                n_regimes=self.config['n_regimes']
            )
            
            self.regime_labels = self.regime_detector.fit(portfolio_returns).predict_regimes(portfolio_returns)
            
            # Log regime characteristics
            characteristics = self.regime_detector.get_regime_characteristics()
            self.logger.info(f"Regime detection completed. Found {self.config['n_regimes']} regimes")
            self.logger.info(f"Regime means: {characteristics['means']}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in regime detection: {str(e)}")
            return False
    
    def estimate_returns(self) -> bool:
        """
        Estimate return distributions using Bayesian methods.
        
        Returns:
            True if estimation successful, False otherwise
        """
        try:
            self.logger.info("Estimating return distributions...")
            
            # Use portfolio-level returns for Bayesian estimation
            portfolio_returns = self.returns_data.mean(axis=1)
            
            self.bayesian_estimator = BayesianReturnEstimator(
                model_type=self.config['bayesian_model_type']
            )
            
            if self.config['bayesian_model_type'] == 'regime_switching':
                self.bayesian_estimator.fit(portfolio_returns, n_regimes=self.config['n_regimes'])
            else:
                self.bayesian_estimator.fit(portfolio_returns)
            
            # Log posterior summary
            posterior_summary = self.bayesian_estimator.get_posterior_summary()
            self.logger.info("Bayesian estimation completed")
            self.logger.info(f"Posterior summary:\n{posterior_summary}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in Bayesian estimation: {str(e)}")
            return False
    
    def optimize_portfolio(self) -> bool:
        """
        Optimize portfolio allocation using the specified method.
        
        Returns:
            True if optimization successful, False otherwise
        """
        try:
            self.logger.info("Optimizing portfolio allocation...")
            
            # Calculate expected returns and covariance
            expected_returns = self.returns_data.mean().values
            covariance_matrix = self.returns_data.cov().values
            
            self.optimizer = PortfolioOptimizer(
                risk_free_rate=self.config['risk_free_rate']
            )
            
            # Apply constraints
            constraints = []
            if self.config['constraints']['long_only']:
                constraints.append(self.optimizer.add_constraints('long_only', n_assets=len(expected_returns)))
            
            if self.config['constraints']['max_weight'] < 1.0:
                constraints.append(self.optimizer.add_constraints('weight_bounds', 
                                                              n_assets=len(expected_returns),
                                                              max_weight=self.config['constraints']['max_weight']))
            
            # Run optimization
            if self.config['optimization_method'] == 'mean_variance':
                result = self.optimizer.mean_variance_optimization(
                    expected_returns, covariance_matrix, 
                    risk_aversion=self.config['risk_aversion'],
                    constraints=constraints
                )
            elif self.config['optimization_method'] == 'risk_parity':
                result = self.optimizer.risk_parity_optimization(
                    covariance_matrix, constraints=constraints
                )
            elif self.config['optimization_method'] == 'max_sharpe':
                result = self.optimizer.maximum_sharpe_optimization(
                    expected_returns, covariance_matrix, constraints=constraints
                )
            elif self.config['optimization_method'] == 'black_litterman':
                market_caps = np.ones(len(expected_returns))
                result = self.optimizer.black_litterman_optimization(
                    market_caps, covariance_matrix
                )
            else:
                raise ValueError(f"Unknown optimization method: {self.config['optimization_method']}")
            
            self.optimization_results = result
            
            # Log optimization results
            self.logger.info("Portfolio optimization completed")
            self.logger.info(f"Portfolio return: {result.get('portfolio_return', 'N/A'):.4f}")
            self.logger.info(f"Portfolio risk: {result.get('portfolio_risk', 'N/A'):.4f}")
            if 'sharpe_ratio' in result:
                self.logger.info(f"Sharpe ratio: {result['sharpe_ratio']:.3f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in portfolio optimization: {str(e)}")
            return False
    
    def evaluate_performance(self) -> bool:
        """
        Calculate comprehensive performance metrics.
        
        Returns:
            True if evaluation successful, False otherwise
        """
        try:
            self.logger.info("Evaluating portfolio performance...")
            
            if self.optimizer is None or self.optimizer.weights is None:
                self.logger.error("No optimization results available for performance evaluation")
                return False
            
            # Calculate portfolio returns
            weights = self.optimizer.weights
            portfolio_returns = (self.returns_data * weights).sum(axis=1)
            
            # Initialize performance calculator
            self.performance_calculator = PerformanceMetrics(
                risk_free_rate=self.config['risk_free_rate']
            )
            
            # Calculate all metrics
            self.performance_metrics = self.performance_calculator.calculate_all_metrics(
                portfolio_returns
            )
            
            # Log key metrics
            self.logger.info("Performance evaluation completed")
            self.logger.info(f"Sharpe ratio: {self.performance_metrics['sharpe_ratio']:.3f}")
            self.logger.info(f"Annualized return: {self.performance_metrics['annualized_return']:.2%}")
            self.logger.info(f"Maximum drawdown: {self.performance_metrics['maximum_drawdown']:.2%}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in performance evaluation: {str(e)}")
            return False
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive analysis report.
        
        Returns:
            Formatted report string
        """
        try:
            report = f"""
{'='*80}
                    BAYESIAN PORTFOLIO OPTIMIZATION REPORT
{'='*80}

ANALYSIS SUMMARY:
{'-'*50}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Assets: {', '.join(self.config['tickers'])}
Data Period: {self.config['start_date']} to {self.config['end_date']}
Number of Regimes: {self.config['n_regimes']}
Bayesian Model: {self.config['bayesian_model_type']}
Optimization Method: {self.config['optimization_method']}

DATA OVERVIEW:
{'-'*50}
Total Observations: {len(self.returns_data)}
Number of Assets: {len(self.returns_data.columns)}
Date Range: {self.returns_data.index[0].strftime('%Y-%m-%d')} to {self.returns_data.index[-1].strftime('%Y-%m-%d')}

REGIME ANALYSIS:
{'-'*50}
"""
            
            if self.regime_detector is not None:
                characteristics = self.regime_detector.get_regime_characteristics()
                for i, (mean, var) in enumerate(zip(characteristics['means'], characteristics['variances'])):
                    report += f"Regime {i}: Mean = {mean:.4f}, Std = {var**0.5:.4f}\n"
            
            report += f"""
PORTFOLIO OPTIMIZATION:
{'-'*50}
"""
            
            if self.optimization_results is not None:
                for key, value in self.optimization_results.items():
                    if isinstance(value, float):
                        report += f"{key.replace('_', ' ').title()}: {value:.4f}\n"
                    else:
                        report += f"{key.replace('_', ' ').title()}: {value}\n"
            
            report += f"""
ASSET ALLOCATION:
{'-'*50}
"""
            
            if self.optimizer is not None and self.optimizer.weights is not None:
                weights_df = self.optimizer.get_weights_summary()
                weights_df['Asset'] = self.config['tickers']
                for _, row in weights_df.iterrows():
                    report += f"{row['Asset']}: {row['Weight_Pct']:.2f}%\n"
            
            report += f"""
PERFORMANCE METRICS:
{'-'*50}
"""
            
            if self.performance_metrics is not None:
                for key, value in self.performance_metrics.items():
                    if isinstance(value, float):
                        if 'ratio' in key.lower():
                            report += f"{key.replace('_', ' ').title()}: {value:.3f}\n"
                        elif 'return' in key.lower() or 'drawdown' in key.lower() or 'var' in key.lower():
                            report += f"{key.replace('_', ' ').title()}: {value:.2%}\n"
                        else:
                            report += f"{key.replace('_', ' ').title()}: {value:.4f}\n"
                    else:
                        report += f"{key.replace('_', ' ').title()}: {value}\n"
            
            report += f"{'='*80}"
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return f"Error generating report: {str(e)}"
    
    def save_results(self, output_dir: str = "results"):
        """
        Save all results to files.
        
        Args:
            output_dir: Directory to save results
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save portfolio data
            if self.portfolio_data is not None:
                self.portfolio_data.to_csv(os.path.join(output_dir, 'portfolio_prices.csv'))
                self.returns_data.to_csv(os.path.join(output_dir, 'portfolio_returns.csv'))
            
            # Save regime labels
            if self.regime_labels is not None:
                import pandas as pd
                regime_df = pd.DataFrame({
                    'date': self.returns_data.index,
                    'regime': self.regime_labels
                })
                regime_df.to_csv(os.path.join(output_dir, 'regime_labels.csv'), index=False)
            
            # Save optimization results
            if self.optimization_results is not None:
                import json
                with open(os.path.join(output_dir, 'optimization_results.json'), 'w') as f:
                    json.dump(self.optimization_results, f, indent=2, default=str)
            
            # Save performance metrics
            if self.performance_metrics is not None:
                import json
                with open(os.path.join(output_dir, 'performance_metrics.json'), 'w') as f:
                    json.dump(self.performance_metrics, f, indent=2, default=str)
            
            # Save report
            report = self.generate_report()
            with open(os.path.join(output_dir, 'analysis_report.txt'), 'w') as f:
                f.write(report)
            
            self.logger.info(f"Results saved to {output_dir}/")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
    
    def run_pipeline(self) -> bool:
        """
        Run the complete portfolio optimization pipeline.
        
        Returns:
            True if pipeline completed successfully, False otherwise
        """
        try:
            self.logger.info("Starting Bayesian Portfolio Optimization pipeline...")
            
            # Step 1: Load data
            if not self.load_data():
                return False
            
            # Step 2: Detect regimes
            if not self.detect_regimes():
                return False
            
            # Step 3: Estimate returns
            if not self.estimate_returns():
                return False
            
            # Step 4: Optimize portfolio
            if not self.optimize_portfolio():
                return False
            
            # Step 5: Evaluate performance
            if not self.evaluate_performance():
                return False
            
            # Generate and display report
            report = self.generate_report()
            print(report)
            
            # Save results
            self.save_results()
            
            self.logger.info("Pipeline completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            return False


def main():
    """Main function to run the portfolio optimization pipeline."""
    parser = argparse.ArgumentParser(description='Bayesian Portfolio Optimization')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--tickers', nargs='+', help='List of asset tickers')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--n-regimes', type=int, help='Number of regimes')
    parser.add_argument('--method', type=str, help='Optimization method')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = BayesianPortfolioOptimizer()
    
    # Override config with command line arguments
    if args.tickers:
        optimizer.config['tickers'] = args.tickers
    if args.start_date:
        optimizer.config['start_date'] = args.start_date
    if args.end_date:
        optimizer.config['end_date'] = args.end_date
    if args.n_regimes:
        optimizer.config['n_regimes'] = args.n_regimes
    if args.method:
        optimizer.config['optimization_method'] = args.method
    
    # Run pipeline
    success = optimizer.run_pipeline()
    
    if success:
        print("\n✅ Portfolio optimization completed successfully!")
        print(f"📁 Results saved to {args.output_dir}/")
    else:
        print("\n❌ Portfolio optimization failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
