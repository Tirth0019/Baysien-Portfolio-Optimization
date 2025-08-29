"""
Data Loader Utility for Financial Data

This module provides utilities for loading financial data from various sources
including Yahoo Finance, CSV files, and APIs.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """
    Utility class for loading financial data from various sources.
    """
    
    def __init__(self):
        """Initialize the data loader."""
        self.logger = logging.getLogger(__name__)
    
    def load_yahoo_data(self, 
                        tickers: List[str], 
                        start_date: str, 
                        end_date: str,
                        interval: str = '1d') -> Optional[pd.DataFrame]:
        """
        Load historical price data from Yahoo Finance.
        
        Args:
            tickers: List of asset ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data interval ('1d', '1wk', '1mo')
            
        Returns:
            DataFrame with historical prices, or None if failed
        """
        try:
            self.logger.info(f"Loading Yahoo Finance data for {len(tickers)} assets...")
            
            data = {}
            for ticker in tickers:
                if ticker.strip():
                    try:
                        ticker_data = yf.download(
                            ticker.strip(), 
                            start=start_date, 
                            end=end_date, 
                            interval=interval,
                            progress=False
                        )
                        
                        if not ticker_data.empty:
                            data[ticker.strip()] = ticker_data['Adj Close']
                            self.logger.debug(f"Loaded data for {ticker}: {len(ticker_data)} observations")
                        else:
                            self.logger.warning(f"No data found for {ticker}")
                            
                    except Exception as e:
                        self.logger.warning(f"Failed to load data for {ticker}: {str(e)}")
                        continue
            
            if data:
                # Create DataFrame and handle missing data
                df = pd.DataFrame(data)
                df = df.dropna()
                
                if len(df) > 0:
                    self.logger.info(f"Successfully loaded data: {len(df)} observations for {len(df.columns)} assets")
                    return df
                else:
                    self.logger.error("No valid data after cleaning")
                    return None
            else:
                self.logger.error("No data could be loaded for any ticker")
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading Yahoo Finance data: {str(e)}")
            return None
    
    def load_csv_data(self, 
                     filepath: str, 
                     date_column: str = 'Date',
                     price_columns: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """
        Load financial data from CSV file.
        
        Args:
            filepath: Path to CSV file
            date_column: Name of the date column
            price_columns: List of price column names (if None, auto-detect)
            
        Returns:
            DataFrame with historical prices, or None if failed
        """
        try:
            self.logger.info(f"Loading CSV data from {filepath}")
            
            # Read CSV file
            df = pd.read_csv(filepath)
            
            if df.empty:
                self.logger.error("CSV file is empty")
                return None
            
            # Set date column as index
            if date_column in df.columns:
                df[date_column] = pd.to_datetime(df[date_column])
                df.set_index(date_column, inplace=True)
            else:
                self.logger.error(f"Date column '{date_column}' not found in CSV")
                return None
            
            # Auto-detect price columns if not specified
            if price_columns is None:
                # Look for common price column patterns
                price_columns = [col for col in df.columns if any(keyword in col.lower() 
                                                               for keyword in ['price', 'close', 'adj_close', 'value'])]
                
                if not price_columns:
                    # If no price columns found, use all numeric columns
                    price_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Filter to only price columns
            if price_columns:
                df = df[price_columns]
            else:
                self.logger.error("No price columns found in CSV")
                return None
            
            # Clean data
            df = df.dropna()
            
            if len(df) > 0:
                self.logger.info(f"Successfully loaded CSV data: {len(df)} observations for {len(df.columns)} assets")
                return df
            else:
                self.logger.error("No valid data after cleaning CSV")
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading CSV data: {str(e)}")
            return None
    
    def load_api_data(self, 
                     api_url: str, 
                     api_key: Optional[str] = None,
                     params: Optional[Dict[str, Any]] = None) -> Optional[pd.DataFrame]:
        """
        Load financial data from API endpoint.
        
        Args:
            api_url: API endpoint URL
            api_key: Optional API key for authentication
            params: Optional query parameters
            
        Returns:
            DataFrame with financial data, or None if failed
        """
        try:
            self.logger.info(f"Loading data from API: {api_url}")
            
            # Prepare headers
            headers = {'Content-Type': 'application/json'}
            if api_key:
                headers['Authorization'] = f'Bearer {api_key}'
            
            # Make API request
            response = requests.get(api_url, headers=headers, params=params)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            # Convert to DataFrame (adjust based on API response structure)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict) and 'data' in data:
                df = pd.DataFrame(data['data'])
            else:
                df = pd.DataFrame(data)
            
            if df.empty:
                self.logger.error("API returned empty data")
                return None
            
            # Clean and process data
            df = self._clean_api_data(df)
            
            if len(df) > 0:
                self.logger.info(f"Successfully loaded API data: {len(df)} observations")
                return df
            else:
                self.logger.error("No valid data after cleaning API response")
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading API data: {str(e)}")
            return None
    
    def _clean_api_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize API data.
        
        Args:
            df: Raw DataFrame from API
            
        Returns:
            Cleaned DataFrame
        """
        try:
            # Remove rows with all NaN values
            df = df.dropna(how='all')
            
            # Convert date columns to datetime
            date_columns = [col for col in df.columns if any(keyword in col.lower() 
                                                           for keyword in ['date', 'time', 'timestamp'])]
            
            for col in date_columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    continue
            
            # Set first date column as index
            if date_columns:
                df.set_index(date_columns[0], inplace=True)
            
            # Convert numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove rows with NaN values in key columns
            df = df.dropna()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error cleaning API data: {str(e)}")
            return df
    
    def get_market_data(self, 
                       tickers: List[str], 
                       start_date: str, 
                       end_date: str,
                       data_source: str = 'yahoo') -> Optional[pd.DataFrame]:
        """
        Get market data from the specified source.
        
        Args:
            tickers: List of asset ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            data_source: Data source ('yahoo', 'csv', 'api')
            
        Returns:
            DataFrame with market data, or None if failed
        """
        if data_source == 'yahoo':
            return self.load_yahoo_data(tickers, start_date, end_date)
        elif data_source == 'csv':
            return self.load_csv_data('data/market_data.csv')
        elif data_source == 'api':
            return self.load_api_data('https://api.example.com/market-data')
        else:
            self.logger.error(f"Unsupported data source: {data_source}")
            return None
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate loaded financial data.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'issues': [],
            'summary': {}
        }
        
        try:
            if df is None or df.empty:
                validation_results['is_valid'] = False
                validation_results['issues'].append("Data is empty or None")
                return validation_results
            
            # Check for missing values
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                validation_results['issues'].append(f"Missing data found: {missing_data.sum()} total missing values")
            
            # Check for infinite values
            infinite_data = np.isinf(df.select_dtypes(include=[np.number])).sum()
            if infinite_data.sum() > 0:
                validation_results['issues'].append(f"Infinite values found: {infinite_data.sum()} total infinite values")
            
            # Check for negative prices
            negative_prices = (df < 0).sum()
            if negative_prices.sum() > 0:
                validation_results['issues'].append(f"Negative prices found: {negative_prices.sum()} total negative values")
            
            # Check data consistency
            if len(df) < 30:
                validation_results['issues'].append("Insufficient data: less than 30 observations")
            
            # Generate summary
            validation_results['summary'] = {
                'total_observations': len(df),
                'total_assets': len(df.columns),
                'date_range': f"{df.index.min()} to {df.index.max()}",
                'missing_data_pct': (missing_data.sum() / (len(df) * len(df.columns))) * 100
            }
            
            # Determine if data is valid
            if validation_results['issues']:
                validation_results['is_valid'] = False
            
            return validation_results
            
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Validation error: {str(e)}")
            return validation_results
    
    def resample_data(self, 
                     df: pd.DataFrame, 
                     frequency: str = 'D',
                     method: str = 'ffill') -> pd.DataFrame:
        """
        Resample data to different frequency.
        
        Args:
            df: Input DataFrame
            frequency: Target frequency ('D', 'W', 'M', 'Q', 'Y')
            method: Method for handling missing values ('ffill', 'bfill', 'interpolate')
            
        Returns:
            Resampled DataFrame
        """
        try:
            self.logger.info(f"Resampling data to {frequency} frequency")
            
            # Resample the data
            resampled = df.resample(frequency).last()
            
            # Handle missing values
            if method == 'ffill':
                resampled = resampled.fillna(method='ffill')
            elif method == 'bfill':
                resampled = resampled.fillna(method='bfill')
            elif method == 'interpolate':
                resampled = resampled.interpolate(method='linear')
            
            # Remove any remaining NaN values
            resampled = resampled.dropna()
            
            self.logger.info(f"Resampling completed: {len(resampled)} observations")
            return resampled
            
        except Exception as e:
            self.logger.error(f"Error resampling data: {str(e)}")
            return df


def load_sample_data() -> pd.DataFrame:
    """
    Load sample data for testing and demonstration.
    
    Returns:
        DataFrame with sample financial data
    """
    # Create sample data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # Generate sample price data
    np.random.seed(42)
    n_assets = 5
    n_days = len(dates)
    
    # Start with random prices
    initial_prices = np.random.uniform(50, 200, n_assets)
    
    # Generate price series with random walk
    prices = np.zeros((n_days, n_assets))
    prices[0] = initial_prices
    
    for i in range(1, n_days):
        # Daily returns with some trend and volatility
        returns = np.random.normal(0.0005, 0.02, n_assets)
        prices[i] = prices[i-1] * (1 + returns)
    
    # Create DataFrame
    df = pd.DataFrame(prices, index=dates, 
                     columns=[f'Asset_{i+1}' for i in range(n_assets)])
    
    return df


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    
    # Load sample data
    sample_data = load_sample_data()
    print("Sample data loaded:")
    print(sample_data.head())
    
    # Validate data
    validation = loader.validate_data(sample_data)
    print("\nValidation results:")
    print(validation)
