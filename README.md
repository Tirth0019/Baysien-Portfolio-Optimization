# Bayesian Portfolio Optimization

A comprehensive Python framework for portfolio optimization using Bayesian methods and regime detection with Hidden Markov Models (HMM).

## 🎯 Overview

This project implements advanced portfolio optimization techniques that combine:
- **Regime Detection**: Hidden Markov Models to identify market states
- **Bayesian Estimation**: PyMC-based return distribution estimation with uncertainty
- **Portfolio Optimization**: Multiple optimization strategies using CVXPY
- **Performance Analysis**: Comprehensive risk and return metrics
- **Interactive Dashboard**: Streamlit-based web interface

## 🏗️ Project Structure

```
bayesian-portfolio-optimizer/
│
├── data/                        # Historical price/fundamental data
│   └── SP500.csv               
│
├── notebooks/                  # Exploratory data analysis, testing
│   └── regime_detection.ipynb
│
├── models/                     # All model scripts
│   ├── bayesian_model.py
│   ├── hmm_model.py
│   └── optimizer.py
│
├── utils/                      # Helper functions
│   ├── data_loader.py
│   └── performance_metrics.py
│
├── app/                        # Streamlit UI
│   └── dashboard.py
│
├── main.py                     # End-to-end runner script
├── requirements.txt
├── config.json
└── .gitignore
```

## 🚀 Features

- **HMM Regime Detection**: Gaussian Hidden Markov Models for market regime identification
- **Bayesian Return Estimation**: Normal, Student-t, and regime-switching models with PyMC
- **Portfolio Optimization**: Mean-variance, risk parity, maximum Sharpe ratio, and Black-Litterman
- **Performance Metrics**: Comprehensive risk and return metrics including Sharpe, Sortino, and Calmar ratios
- **Data Management**: Yahoo Finance integration, CSV and API data loading
- **Interactive Dashboard**: Streamlit-based web interface for real-time analysis

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Tirth0019/Baysien-Portfolio-Optimization.git
   cd Baysien-Portfolio-Optimization
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

### Quick Start

1. **Run the main pipeline**
   ```bash
   python main.py --tickers AAPL GOOGL MSFT --n-regimes 3 --method max_sharpe
   ```

2. **Launch the dashboard**
   ```bash
   streamlit run app/dashboard.py
   ```

3. **Explore with Jupyter**
   ```bash
   jupyter notebook notebooks/regime_detection.ipynb
   ```

### Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --tickers TEXT...        List of asset tickers
  --start-date TEXT        Start date (YYYY-MM-DD)
  --end-date TEXT          End date (YYYY-MM-DD)
  --n-regimes INTEGER      Number of regimes to detect
  --method TEXT            Optimization method
  --output-dir TEXT        Output directory for results
  --config TEXT            Path to configuration file
```

### Configuration

The system supports configuration files in JSON format. See `config.json` for default settings.

## 📊 Example Usage

```python
from models.hmm_model import HMMRegimeDetector
from models.bayesian_model import BayesianReturnEstimator
from models.optimizer import PortfolioOptimizer
from utils.data_loader import DataLoader

# Load data
loader = DataLoader()
data = loader.load_yahoo_data(['AAPL', 'GOOGL'], '2020-01-01', '2023-12-31')
returns = data.pct_change().dropna()

# Detect regimes
hmm = HMMRegimeDetector(n_regimes=3)
regime_labels = hmm.fit(returns.mean(axis=1)).predict_regimes(returns.mean(axis=1))

# Estimate returns
estimator = BayesianReturnEstimator(model_type="student_t")
estimator.fit(returns.mean(axis=1))

# Optimize portfolio
optimizer = PortfolioOptimizer(risk_free_rate=0.02)
result = optimizer.maximum_sharpe_optimization(
    expected_returns=returns.mean().values,
    covariance_matrix=returns.cov().values
)
```

## 📚 Documentation

### Repository Links
- **Source Code**: [GitHub Repository](https://github.com/Tirth0019/Baysien-Portfolio-Optimization)
- **Issues**: [GitHub Issues](https://github.com/Tirth0019/Baysien-Portfolio-Optimization/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Tirth0019/Baysien-Portfolio-Optimization/discussions)

### Code Structure
- **Models**: Core algorithms in the `models/` directory
- **Utilities**: Helper functions in the `utils/` directory  
- **Dashboard**: Streamlit interface in the `app/` directory
- **Examples**: Sample data and notebooks for testing

## 📄 License

This project is licensed under the MIT License.

---

**Made with ❤️ by the Bayesian Portfolio Optimization Team**
