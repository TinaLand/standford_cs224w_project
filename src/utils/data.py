"""
Data Utilities for CS224W Stock RL GNN Project
==============================================
Helper functions for data cleaning, normalization, and preprocessing.

This module provides common utility functions used across different phases:
- Data validation and cleaning
- Normalization and standardization
- Date handling utilities
- File I/O helpers
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from typing import Union, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Set up paths
BASE_DIR = Path(__file__).parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
DATA_EDGES_DIR = BASE_DIR / "data" / "edges"

def validate_data_directories():
    """
    Validate that all required data directories exist and create them if they don't.
    
    Returns:
        bool: True if all directories are ready, False otherwise
    """
    directories = [DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_EDGES_DIR]
    
    for directory in directories:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            print(f" Created directory: {directory}")
        else:
            print(f" Directory exists: {directory}")
    
    return True

def clean_dataframe(df: pd.DataFrame, 
                   remove_duplicates: bool = True,
                   handle_missing: str = 'ffill',
                   missing_threshold: float = 0.5) -> pd.DataFrame:
    """
    Clean a DataFrame by handling missing values, duplicates, and outliers.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        remove_duplicates (bool): Whether to remove duplicate rows
        handle_missing (str): Method to handle missing values ('ffill', 'bfill', 'drop', 'interpolate')
        missing_threshold (float): Drop columns with more than this fraction of missing values
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    print(f" Cleaning DataFrame with shape: {df.shape}")
    
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Remove columns with too many missing values
    missing_fraction = df_clean.isnull().sum() / len(df_clean)
    columns_to_drop = missing_fraction[missing_fraction > missing_threshold].index.tolist()
    
    if columns_to_drop:
        df_clean = df_clean.drop(columns=columns_to_drop)
        print(f"  - Dropped {len(columns_to_drop)} columns with >{missing_threshold*100}% missing values")
    
    # Handle missing values
    if handle_missing == 'ffill':
        df_clean = df_clean.ffill()
    elif handle_missing == 'bfill':
        df_clean = df_clean.bfill()
    elif handle_missing == 'interpolate':
        df_clean = df_clean.interpolate()
    elif handle_missing == 'drop':
        df_clean = df_clean.dropna()
    
    print(f"  - Handled missing values using method: {handle_missing}")
    
    # Remove duplicates
    if remove_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed_duplicates = initial_rows - len(df_clean)
        if removed_duplicates > 0:
            print(f"  - Removed {removed_duplicates} duplicate rows")
    
    print(f" Cleaned DataFrame shape: {df_clean.shape}")
    return df_clean

def normalize_features(df: pd.DataFrame, 
                      columns: Optional[List[str]] = None,
                      method: str = 'standard',
                      save_scaler: bool = False,
                      scaler_path: Optional[str] = None) -> Tuple[pd.DataFrame, object]:
    """
    Normalize numerical features in a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (List[str], optional): Columns to normalize. If None, normalize all numeric columns
        method (str): Normalization method ('standard', 'minmax', 'robust')
        save_scaler (bool): Whether to save the fitted scaler
        scaler_path (str, optional): Path to save the scaler
    
    Returns:
        Tuple[pd.DataFrame, object]: Normalized DataFrame and fitted scaler
    """
    print(f" Normalizing features using method: {method}")
    
    df_normalized = df.copy()
    
    # Select columns to normalize
    if columns is None:
        columns = df_normalized.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"  - Normalizing {len(columns)} columns")
    
    # Choose scaler
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    
    # Fit and transform
    df_normalized[columns] = scaler.fit_transform(df_normalized[columns])
    
    # Save scaler if requested
    if save_scaler and scaler_path:
        import joblib
        joblib.dump(scaler, scaler_path)
        print(f"  - Saved scaler to: {scaler_path}")
    
    print(f" Normalized {len(columns)} features")
    return df_normalized, scaler

def calculate_returns(price_series: pd.Series, 
                     method: str = 'simple',
                     periods: int = 1) -> pd.Series:
    """
    Calculate returns from a price series.
    
    Args:
        price_series (pd.Series): Price data
        method (str): Return calculation method ('simple', 'log')
        periods (int): Number of periods for return calculation
    
    Returns:
        pd.Series: Calculated returns
    """
    if method == 'simple':
        returns = price_series.pct_change(periods=periods)
    elif method == 'log':
        returns = np.log(price_series / price_series.shift(periods))
    else:
        raise ValueError(f"Unsupported return method: {method}")
    
    return returns.dropna()

def resample_data(df: pd.DataFrame, 
                 date_column: str,
                 freq: str,
                 agg_method: Union[str, dict] = 'last') -> pd.DataFrame:
    """
    Resample time series data to a different frequency.
    
    Args:
        df (pd.DataFrame): Input DataFrame with datetime index or column
        date_column (str): Name of the date column (if not index)
        freq (str): Target frequency ('D', 'W', 'M', 'Q', 'Y')
        agg_method (str or dict): Aggregation method ('last', 'first', 'mean', 'sum') or dict for column-specific methods
    
    Returns:
        pd.DataFrame: Resampled DataFrame
    """
    print(f" Resampling data to frequency: {freq}")
    
    df_resampled = df.copy()
    
    # Set date column as index if needed
    if date_column in df_resampled.columns:
        df_resampled[date_column] = pd.to_datetime(df_resampled[date_column])
        df_resampled = df_resampled.set_index(date_column)
    
    # Resample
    if isinstance(agg_method, str):
        if agg_method == 'last':
            df_resampled = df_resampled.resample(freq).last()
        elif agg_method == 'first':
            df_resampled = df_resampled.resample(freq).first()
        elif agg_method == 'mean':
            df_resampled = df_resampled.resample(freq).mean()
        elif agg_method == 'sum':
            df_resampled = df_resampled.resample(freq).sum()
    else:
        df_resampled = df_resampled.resample(freq).agg(agg_method)
    
    print(f" Resampled from {len(df)} to {len(df_resampled)} rows")
    return df_resampled

def detect_outliers(df: pd.DataFrame, 
                   columns: Optional[List[str]] = None,
                   method: str = 'iqr',
                   threshold: float = 1.5) -> pd.DataFrame:
    """
    Detect outliers in numerical columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (List[str], optional): Columns to check for outliers
        method (str): Outlier detection method ('iqr', 'zscore')
        threshold (float): Threshold for outlier detection
    
    Returns:
        pd.DataFrame: Boolean DataFrame indicating outliers
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outliers = pd.DataFrame(False, index=df.index, columns=df.columns)
    
    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers[col] = z_scores > threshold
    
    outlier_counts = outliers[columns].sum()
    print(f" Detected outliers by column:")
    for col, count in outlier_counts.items():
        if count > 0:
            print(f"  - {col}: {count} outliers ({count/len(df)*100:.1f}%)")
    
    return outliers

def save_processed_data(df: pd.DataFrame, 
                       filename: str,
                       directory: str = 'processed',
                       include_timestamp: bool = False) -> str:
    """
    Save processed data to the appropriate directory.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        filename (str): Name of the file (without extension)
        directory (str): Directory to save to ('raw', 'processed', 'edges')
        include_timestamp (bool): Whether to include timestamp in filename
    
    Returns:
        str: Full path of saved file
    """
    # Choose directory
    if directory == 'raw':
        save_dir = DATA_RAW_DIR
    elif directory == 'processed':
        save_dir = DATA_PROCESSED_DIR
    elif directory == 'edges':
        save_dir = DATA_EDGES_DIR
    else:
        save_dir = Path(directory)
    
    # Add timestamp if requested
    if include_timestamp:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{filename}_{timestamp}"
    
    # Ensure .csv extension
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    # Save file
    file_path = save_dir / filename
    df.to_csv(file_path, index=True)
    
    print(f" Saved data to: {file_path}")
    print(f" Data shape: {df.shape}")
    
    return str(file_path)

def load_data_file(filename: str, 
                   directory: str = 'processed',
                   parse_dates: bool = True,
                   index_col: Optional[Union[int, str]] = None) -> pd.DataFrame:
    """
    Load data from the project directory structure.
    
    Args:
        filename (str): Name of the file to load
        directory (str): Directory to load from ('raw', 'processed', 'edges')
        parse_dates (bool): Whether to parse date columns
        index_col (int or str, optional): Column to use as index
    
    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    # Choose directory
    if directory == 'raw':
        load_dir = DATA_RAW_DIR
    elif directory == 'processed':
        load_dir = DATA_PROCESSED_DIR
    elif directory == 'edges':
        load_dir = DATA_EDGES_DIR
    else:
        load_dir = Path(directory)
    
    # Load file
    file_path = load_dir / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load with appropriate settings
    df = pd.read_csv(file_path, parse_dates=parse_dates, index_col=index_col)
    
    print(f" Loaded data from: {file_path}")
    print(f" Data shape: {df.shape}")
    
    return df

def validate_ticker_data(df: pd.DataFrame, 
                        required_columns: List[str],
                        min_observations: int = 100) -> dict:
    """
    Validate ticker data quality and completeness.
    
    Args:
        df (pd.DataFrame): DataFrame with ticker data
        required_columns (List[str]): Columns that must be present
        min_observations (int): Minimum number of observations per ticker
    
    Returns:
        dict: Validation results with statistics and warnings
    """
    results = {
        'total_tickers': 0,
        'valid_tickers': [],
        'invalid_tickers': [],
        'missing_columns': [],
        'warnings': []
    }
    
    # Check required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    results['missing_columns'] = missing_cols
    
    if missing_cols:
        results['warnings'].append(f"Missing required columns: {missing_cols}")
    
    # Check ticker data if ticker column exists
    if 'ticker' in df.columns:
        ticker_counts = df['ticker'].value_counts()
        results['total_tickers'] = len(ticker_counts)
        
        # Identify valid and invalid tickers
        valid_tickers = ticker_counts[ticker_counts >= min_observations].index.tolist()
        invalid_tickers = ticker_counts[ticker_counts < min_observations].index.tolist()
        
        results['valid_tickers'] = valid_tickers
        results['invalid_tickers'] = invalid_tickers
        
        if invalid_tickers:
            results['warnings'].append(f"Tickers with insufficient data (<{min_observations} obs): {len(invalid_tickers)}")
    
    # Print summary
    print(" Data Validation Results:")
    print(f"  - Total tickers: {results['total_tickers']}")
    print(f"  - Valid tickers: {len(results['valid_tickers'])}")
    print(f"  - Invalid tickers: {len(results['invalid_tickers'])}")
    
    if results['warnings']:
        print("  Warnings:")
        for warning in results['warnings']:
            print(f"    - {warning}")
    
    return results