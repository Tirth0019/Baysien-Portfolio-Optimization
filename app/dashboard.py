 """
Streamlit Dashboard for Bayesian Portfolio Optimization

This module provides an interactive web interface for portfolio optimization
with sliders, plots, and regime timeline visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from hmm_model import HMMRegimeDetector
from bayesian_model import BayesianReturnEstimator
from optimizer import PortfolioOptimizer
from performance_metrics import PerformanceMetrics


class PortfolioDashboard:
    """
    Streamlit dashboard for portfolio optimization and analysis.
    """
    
    def __init__(self):
        """Initialize the dashboard."""
        st.set_page_config(
            page_title="Bayesian Portfolio Optimization",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize session state
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'portfolio_data' not in st.session_state:
            st.session_state.portfolio_data = None
        if 'regime_model' not in st.session_state:
            st.session_state.regime_model = None
        if 'bayesian_model' not in st.session_state:
            st.session_state.bayesian_model = None
        if 'optimizer' not in st.session_state:
            st.session_state.optimizer = None
        
    def main(self):
        """Main dashboard function."""
        st.title("📊 Bayesian Portfolio Optimization Dashboard")
        st.markdown("---")
        
        # Sidebar
        self.sidebar()
        
        # Main content
        if st.session_state.data_loaded:
            self.main_content()
        else:
            self.welcome_screen()
    
    def sidebar(self):
        """Create the sidebar with controls."""
        st.sidebar.header("🔧 Controls")
        
        # Data loading section
        st.sidebar.subheader("📈 Data Loading")
        
        # Asset selection
        default_tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        tickers = st.sidebar.text_area(
            "Asset Tickers (one per line):",
            value="\n".join(default_tickers),
            height=100
        ).strip().split('\n')
        
        # Date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)
        
        start_date = st.sidebar.date_input(
            "Start Date:",
            value=start_date,
            max_value=end_date
        )
        
        end_date = st.sidebar.date_input(
            "End Date:",
            value=end_date,
            max_value=end_date
        )
        
        # Load data button
        if st.sidebar.button("🚀 Load Data", type="primary"):
            self.load_data(tickers, start_date, end_date)
        
        if st.session_state.data_loaded:
            st.sidebar.markdown("---")
            
            # Model parameters
            st.sidebar.subheader("🤖 Model Parameters")
            
            # HMM parameters
            n_regimes = st.sidebar.slider("Number of Regimes:", 2, 5, 3)
            
            # Bayesian model type
            bayesian_model_type = st.sidebar.selectbox(
                "Bayesian Model Type:",
                ["normal", "student_t", "regime_switching"]
            )
            
            # Optimization method
            optimization_method = st.sidebar.selectbox(
                "Optimization Method:",
                ["mean_variance", "risk_parity", "max_sharpe", "black_litterman"]
            )
            
            # Risk aversion
            risk_aversion = st.sidebar.slider("Risk Aversion:", 0.1, 5.0, 1.0, 0.1)
            
            # Risk-free rate
            risk_free_rate = st.sidebar.slider("Risk-Free Rate (%):", 0.0, 10.0, 2.0, 0.1) / 100
            
            # Run analysis button
            if st.sidebar.button("🔍 Run Analysis", type="primary"):
                self.run_analysis(n_regimes, bayesian_model_type, optimization_method, risk_aversion, risk_free_rate)
    
    def welcome_screen(self):
        """Display welcome screen when no data is loaded."""
        st.markdown("""
        ## Welcome to Bayesian Portfolio Optimization! 🎯
        
        This dashboard helps you optimize your portfolio using advanced Bayesian methods and regime detection.
        
        ### Features:
        - **Regime Detection**: Identify market states using Hidden Markov Models
        - **Bayesian Estimation**: Estimate return distributions with uncertainty
        - **Portfolio Optimization**: Optimize allocations using various methods
        - **Performance Analysis**: Comprehensive risk and return metrics
        
        ### Getting Started:
        1. Enter asset tickers in the sidebar
        2. Select date range
        3. Click "Load Data" to fetch market data
        4. Configure model parameters
        5. Run the analysis
        """)
        
        # Example data
        st.subheader("📊 Example Portfolio")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Sample Assets:**
            - AAPL (Apple)
            - GOOGL (Google)
            - MSFT (Microsoft)
            - AMZN (Amazon)
            - TSLA (Tesla)
            """)
        
        with col2:
            st.markdown("""
            **Sample Analysis:**
            - 2-year historical data
            - 3-regime HMM model
            - Student-t Bayesian model
            - Maximum Sharpe optimization
            """)
    
    def load_data(self, tickers, start_date, end_date):
        """Load market data for the specified tickers."""
        try:
            with st.spinner("📥 Loading market data..."):
                # Download data
                data = {}
                for ticker in tickers:
                    if ticker.strip():
                        ticker_data = yf.download(ticker.strip(), start=start_date, end=end_date, progress=False)
                        if not ticker_data.empty:
                            data[ticker.strip()] = ticker_data['Adj Close']
                
                if data:
                    # Create DataFrame
                    df = pd.DataFrame(data)
                    df = df.dropna()
                    
                    # Store in session state
                    st.session_state.portfolio_data = df
                    st.session_state.data_loaded = True
                    
                    st.success(f"✅ Data loaded successfully! {len(df)} observations for {len(df.columns)} assets.")
                else:
                    st.error("❌ No data could be loaded. Please check ticker symbols.")
                    
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
    
    def run_analysis(self, n_regimes, bayesian_model_type, optimization_method, risk_aversion, risk_free_rate):
        """Run the complete portfolio analysis."""
        try:
            with st.spinner("🔍 Running analysis..."):
                data = st.session_state.portfolio_data
                returns = data.pct_change().dropna()
                
                # 1. Regime Detection
                st.subheader("🎭 Regime Detection")
                regime_labels, regime_model = HMMRegimeDetector(n_regimes=n_regimes).fit(returns.mean(axis=1))
                st.session_state.regime_model = regime_model
                
                # Display regime characteristics
                characteristics = regime_model.get_regime_characteristics()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Regime 0 Mean", f"{characteristics['means'][0]:.4f}")
                with col2:
                    st.metric("Regime 1 Mean", f"{characteristics['means'][1]:.4f}")
                with col3:
                    if n_regimes > 2:
                        st.metric("Regime 2 Mean", f"{characteristics['means'][2]:.4f}")
                
                # 2. Bayesian Estimation
                st.subheader("🔮 Bayesian Return Estimation")
                bayesian_estimator = BayesianReturnEstimator(model_type=bayesian_model_type)
                
                if bayesian_model_type == "regime_switching":
                    bayesian_estimator.fit(returns.mean(axis=1), n_regimes=n_regimes)
                else:
                    bayesian_estimator.fit(returns.mean(axis=1))
                
                st.session_state.bayesian_model = bayesian_estimator
                
                # Display posterior summary
                posterior_summary = bayesian_estimator.get_posterior_summary()
                st.dataframe(posterior_summary)
                
                # 3. Portfolio Optimization
                st.subheader("⚡ Portfolio Optimization")
                optimizer = PortfolioOptimizer(risk_free_rate=risk_free_rate)
                
                # Calculate expected returns and covariance
                expected_returns = returns.mean().values
                covariance_matrix = returns.cov().values
                
                if optimization_method == "mean_variance":
                    result = optimizer.mean_variance_optimization(
                        expected_returns, covariance_matrix, risk_aversion=risk_aversion
                    )
                elif optimization_method == "risk_parity":
                    result = optimizer.risk_parity_optimization(covariance_matrix)
                elif optimization_method == "max_sharpe":
                    result = optimizer.maximum_sharpe_optimization(expected_returns, covariance_matrix)
                elif optimization_method == "black_litterman":
                    market_caps = np.ones(len(expected_returns))  # Equal weight prior
                    result = optimizer.black_litterman_optimization(market_caps, covariance_matrix)
                
                st.session_state.optimizer = optimizer
                
                # Display optimization results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Portfolio Return", f"{result['portfolio_return']:.4f}")
                with col2:
                    st.metric("Portfolio Risk", f"{result['portfolio_risk']:.4f}")
                with col3:
                    if 'sharpe_ratio' in result:
                        st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.3f}")
                
                # Display weights
                weights_df = optimizer.get_weights_summary()
                weights_df['Asset'] = data.columns
                st.dataframe(weights_df)
                
                st.success("✅ Analysis completed successfully!")
                
        except Exception as e:
            st.error(f"❌ Error running analysis: {str(e)}")
    
    def main_content(self):
        """Display main content after data is loaded."""
        if not st.session_state.data_loaded:
            return
        
        data = st.session_state.portfolio_data
        
        # Data overview
        st.subheader("📊 Data Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Number of Assets", len(data.columns))
        with col2:
            st.metric("Number of Observations", len(data))
        with col3:
            st.metric("Date Range", f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        
        # Price chart
        st.subheader("📈 Asset Prices")
        fig = go.Figure()
        
        for asset in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[asset],
                mode='lines',
                name=asset,
                line=dict(width=1)
            ))
        
        fig.update_layout(
            title="Asset Price Evolution",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Returns analysis
        st.subheader("📊 Returns Analysis")
        returns = data.pct_change().dropna()
        
        # Returns distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                returns.melt(),
                x='value',
                color='variable',
                title="Return Distribution by Asset",
                barmode='overlay',
                opacity=0.7
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Correlation heatmap
            corr_matrix = returns.corr()
            fig = px.imshow(
                corr_matrix,
                title="Asset Correlation Matrix",
                color_continuous_scale='RdBu',
                aspect='auto'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Regime analysis (if available)
        if st.session_state.regime_model is not None:
            st.subheader("🎭 Regime Analysis")
            
            regime_labels, _ = HMMRegimeDetector(n_regimes=st.session_state.regime_model.n_regimes).fit(returns.mean(axis=1))
            
            # Regime timeline
            fig = go.Figure()
            
            for regime in range(st.session_state.regime_model.n_regimes):
                mask = regime_labels == regime
                fig.add_trace(go.Scatter(
                    x=returns.index[mask],
                    y=returns.mean(axis=1)[mask],
                    mode='markers',
                    name=f'Regime {regime}',
                    marker=dict(size=8, opacity=0.7)
                ))
            
            fig.update_layout(
                title="Market Regimes Over Time",
                xaxis_title="Date",
                yaxis_title="Portfolio Returns",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics (if optimizer available)
        if st.session_state.optimizer is not None:
            st.subheader("📊 Performance Metrics")
            
            # Calculate portfolio returns
            weights = st.session_state.optimizer.weights
            portfolio_returns = (returns * weights).sum(axis=1)
            
            # Performance calculator
            perf_calc = PerformanceMetrics(risk_free_rate=st.session_state.optimizer.risk_free_rate)
            metrics = perf_calc.calculate_all_metrics(portfolio_returns)
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}")
                st.metric("Annualized Return", f"{metrics['annualized_return']:.2%}")
            
            with col2:
                st.metric("Annualized Volatility", f"{metrics['annualized_volatility']:.2%}")
                st.metric("Maximum Drawdown", f"{metrics['maximum_drawdown']:.2%}")
            
            with col3:
                st.metric("Sortino Ratio", f"{metrics['sortino_ratio']:.3f}")
                st.metric("Calmar Ratio", f"{metrics['calmar_ratio']:.3f}")
            
            with col4:
                st.metric("Value at Risk (95%)", f"{metrics['var_95']:.2%}")
                st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
            
            # Performance charts
            st.subheader("📈 Performance Charts")
            
            # Cumulative returns
            cumulative_returns = perf_calc.cumulative_returns(portfolio_returns)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns.values,
                mode='lines',
                name='Portfolio',
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(
                title="Cumulative Portfolio Returns",
                xaxis_title="Date",
                yaxis_title="Cumulative Return",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Drawdown
            drawdown = perf_calc.drawdown_series(portfolio_returns)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.3)',
                line=dict(color='red'),
                name='Drawdown'
            ))
            
            fig.update_layout(
                title="Portfolio Drawdown",
                xaxis_title="Date",
                yaxis_title="Drawdown",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)


def main():
    """Main function to run the dashboard."""
    dashboard = PortfolioDashboard()
    dashboard.main()


if __name__ == "__main__":
    main()
