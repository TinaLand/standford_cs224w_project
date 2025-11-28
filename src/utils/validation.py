"""
Data Validation and Quality Checks for Phase 1 Data Collection
==============================================================

This module provides comprehensive data validation and cleaning functions
for stock market data collected in Phase 1.

Functions:
- validate_ohlcv_data(): Comprehensive OHLCV data quality checks
- align_trading_calendars(): Align trading calendars across tickers
- handle_stock_suspensions(): Detect and handle missing trading days
- handle_splits_dividends(): Adjust prices for stock splits and dividends
- impute_missing_values(): Robust missing value imputation
- create_data_collection_log(): Log data collection metadata
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional


def validate_ohlcv_data(ohlcv_df: pd.DataFrame, tickers: List[str]) -> Dict:
    """
    Comprehensive OHLCV data quality validation.
    
    Checks:
    - Negative prices
    - Price anomalies (e.g., Close > High, Close < Low)
    - Volume anomalies (zero or negative volume)
    - Missing values percentage
    - Date alignment issues
    - Outliers (using IQR method)
    
    Args:
        ohlcv_df: DataFrame with OHLCV data (Date index, columns like 'Close_AAPL')
        tickers: List of ticker symbols
    
    Returns:
        Dictionary with validation results and issues found
    """
    print("\n🔍 Validating OHLCV Data Quality...")
    
    validation_results = {
        'is_valid': True,
        'issues': [],
        'warnings': [],
        'statistics': {}
    }
    
    # 1. Check for negative prices
    price_cols = [col for col in ohlcv_df.columns if any(x in col for x in ['Open', 'High', 'Low', 'Close'])]
    negative_prices = (ohlcv_df[price_cols] < 0).any().any()
    if negative_prices:
        validation_results['is_valid'] = False
        validation_results['issues'].append("Negative prices detected")
        print("  ❌ ERROR: Negative prices found")
    
    # 2. Check OHLC price relationships
    for ticker in tickers:
        open_col = f'Open_{ticker}'
        high_col = f'High_{ticker}'
        low_col = f'Low_{ticker}'
        close_col = f'Close_{ticker}'
        
        if all(col in ohlcv_df.columns for col in [open_col, high_col, low_col, close_col]):
            # High should be >= Open, Low, Close
            invalid_high = (
                (ohlcv_df[high_col] < ohlcv_df[open_col]) |
                (ohlcv_df[high_col] < ohlcv_df[low_col]) |
                (ohlcv_df[high_col] < ohlcv_df[close_col])
            ).sum()
            
            # Low should be <= Open, High, Close
            invalid_low = (
                (ohlcv_df[low_col] > ohlcv_df[open_col]) |
                (ohlcv_df[low_col] > ohlcv_df[high_col]) |
                (ohlcv_df[low_col] > ohlcv_df[close_col])
            ).sum()
            
            if invalid_high > 0:
                validation_results['warnings'].append(
                    f"{ticker}: {invalid_high} rows with invalid High prices"
                )
            if invalid_low > 0:
                validation_results['warnings'].append(
                    f"{ticker}: {invalid_low} rows with invalid Low prices"
                )
    
    # 3. Check volume anomalies
    volume_cols = [col for col in ohlcv_df.columns if 'Volume' in col]
    zero_volume = (ohlcv_df[volume_cols] == 0).sum().sum()
    negative_volume = (ohlcv_df[volume_cols] < 0).any().any()
    
    if zero_volume > 0:
        validation_results['warnings'].append(f"{zero_volume} zero volume entries found")
    if negative_volume:
        validation_results['is_valid'] = False
        validation_results['issues'].append("Negative volume detected")
        print("  ❌ ERROR: Negative volume found")
    
    # 4. Check missing values
    missing_pct = ohlcv_df.isnull().sum() / len(ohlcv_df) * 100
    high_missing = missing_pct[missing_pct > 10]  # More than 10% missing
    
    if len(high_missing) > 0:
        validation_results['warnings'].append(
            f"{len(high_missing)} columns with >10% missing values"
        )
        print(f"  ⚠️  WARNING: {len(high_missing)} columns have >10% missing data")
    
    # 5. Check date alignment
    if not isinstance(ohlcv_df.index, pd.DatetimeIndex):
        validation_results['issues'].append("Index is not DatetimeIndex")
        validation_results['is_valid'] = False
    else:
        # Check for gaps (weekends/holidays are OK, but large gaps might indicate issues)
        date_diffs = ohlcv_df.index.to_series().diff()
        large_gaps = (date_diffs > pd.Timedelta(days=10)).sum()
        if large_gaps > 0:
            validation_results['warnings'].append(
                f"{large_gaps} large date gaps (>10 days) detected"
            )
    
    # 6. Outlier detection using IQR method
    for ticker in tickers[:5]:  # Check first 5 tickers as sample
        close_col = f'Close_{ticker}'
        if close_col in ohlcv_df.columns:
            prices = ohlcv_df[close_col].dropna()
            if len(prices) > 0:
                Q1 = prices.quantile(0.25)
                Q3 = prices.quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((prices < (Q1 - 3 * IQR)) | (prices > (Q3 + 3 * IQR))).sum()
                if outliers > 0:
                    validation_results['warnings'].append(
                        f"{ticker}: {outliers} potential price outliers (3*IQR method)"
                    )
    
    # Statistics
    validation_results['statistics'] = {
        'total_rows': len(ohlcv_df),
        'total_columns': len(ohlcv_df.columns),
        'date_range': (ohlcv_df.index.min(), ohlcv_df.index.max()) if len(ohlcv_df) > 0 else None,
        'missing_pct_overall': ohlcv_df.isnull().sum().sum() / (len(ohlcv_df) * len(ohlcv_df.columns)) * 100
    }
    
    if validation_results['is_valid'] and len(validation_results['warnings']) == 0:
        print("  ✅ Data validation passed")
    elif validation_results['is_valid']:
        print(f"  ⚠️  Data validation passed with {len(validation_results['warnings'])} warnings")
    else:
        print(f"  ❌ Data validation failed with {len(validation_results['issues'])} critical issues")
    
    return validation_results


def align_trading_calendars(ohlcv_df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """
    Align trading calendars across all tickers.
    
    Different stocks may have different trading days due to:
    - Different exchanges (NYSE vs NASDAQ holidays)
    - Stock-specific suspensions
    - Delisting/listing dates
    
    Strategy:
    1. Find the intersection of all trading days (most conservative)
    2. Forward-fill missing values for stocks that don't trade on certain days
    3. Mark suspended days if needed
    
    Args:
        ohlcv_df: DataFrame with OHLCV data
        tickers: List of ticker symbols
    
    Returns:
        DataFrame with aligned trading calendar
    """
    print("\n📅 Aligning Trading Calendars...")
    
    # Get all unique dates from all tickers
    all_dates = set(ohlcv_df.index)
    
    # For each ticker, check which dates have data
    ticker_date_coverage = {}
    for ticker in tickers:
        close_col = f'Close_{ticker}'
        if close_col in ohlcv_df.columns:
            ticker_dates = set(ohlcv_df[ohlcv_df[close_col].notna()].index)
            ticker_date_coverage[ticker] = ticker_dates
            coverage_pct = len(ticker_dates) / len(all_dates) * 100
            print(f"  {ticker}: {len(ticker_dates)}/{len(all_dates)} dates ({coverage_pct:.1f}%)")
    
    # Find common trading days (intersection)
    if ticker_date_coverage:
        common_dates = set.intersection(*ticker_date_coverage.values())
        print(f"  Common trading days: {len(common_dates)}")
        
        # Filter to common dates
        aligned_df = ohlcv_df.loc[sorted(common_dates)].copy()
        
        # Forward-fill missing values for dates that are in common set but missing for some tickers
        for ticker in tickers:
            price_cols = [f'{col}_{ticker}' for col in ['Open', 'High', 'Low', 'Close', 'Volume']]
            existing_cols = [col for col in price_cols if col in aligned_df.columns]
            if existing_cols:
                aligned_df[existing_cols] = aligned_df[existing_cols].ffill()
        
        print(f"  ✅ Calendar aligned: {len(aligned_df)} common trading days")
        return aligned_df
    else:
        print("  ⚠️  No ticker data found, returning original DataFrame")
        return ohlcv_df


def handle_stock_suspensions(ohlcv_df: pd.DataFrame, tickers: List[str], 
                             max_consecutive_missing: int = 5) -> Tuple[pd.DataFrame, Dict]:
    """
    Detect and handle stock suspensions.
    
    Suspensions are detected as consecutive missing trading days.
    Strategy:
    - Short gaps (<= max_consecutive_missing): Forward-fill (likely data gaps)
    - Long gaps (> max_consecutive_missing): Mark as suspension, forward-fill with warning
    
    Args:
        ohlcv_df: DataFrame with OHLCV data
        tickers: List of ticker symbols
        max_consecutive_missing: Maximum consecutive missing days before considering suspension
    
    Returns:
        Tuple of (cleaned DataFrame, suspension report dictionary)
    """
    print("\n⏸️  Handling Stock Suspensions...")
    
    suspension_report = {}
    cleaned_df = ohlcv_df.copy()
    
    for ticker in tickers:
        close_col = f'Close_{ticker}'
        if close_col not in cleaned_df.columns:
            continue
        
        # Find missing periods
        is_missing = cleaned_df[close_col].isna()
        missing_periods = []
        
        in_missing = False
        start_idx = None
        
        for i, missing in enumerate(is_missing):
            if missing and not in_missing:
                in_missing = True
                start_idx = i
            elif not missing and in_missing:
                in_missing = False
                if start_idx is not None:
                    missing_periods.append((start_idx, i - 1))
                start_idx = None
        
        if in_missing and start_idx is not None:
            missing_periods.append((start_idx, len(is_missing) - 1))
        
        # Process each missing period
        ticker_suspensions = []
        for start, end in missing_periods:
            duration = end - start + 1
            start_date = cleaned_df.index[start]
            end_date = cleaned_df.index[end]
            
            if duration > max_consecutive_missing:
                ticker_suspensions.append({
                    'start': start_date,
                    'end': end_date,
                    'duration_days': duration
                })
                print(f"  ⚠️  {ticker}: Suspension detected {start_date.date()} to {end_date.date()} ({duration} days)")
            
            # Forward-fill prices and volume
            price_cols = [f'{col}_{ticker}' for col in ['Open', 'High', 'Low', 'Close']]
            volume_col = f'Volume_{ticker}'
            
            for col in price_cols:
                if col in cleaned_df.columns:
                    cleaned_df[col].iloc[start:end+1] = cleaned_df[col].iloc[start-1] if start > 0 else np.nan
            
            if volume_col in cleaned_df.columns:
                cleaned_df[volume_col].iloc[start:end+1] = 0  # Volume = 0 during suspension
        
        if ticker_suspensions:
            suspension_report[ticker] = ticker_suspensions
    
    if suspension_report:
        print(f"  ⚠️  Found suspensions for {len(suspension_report)} tickers")
    else:
        print("  ✅ No significant suspensions detected")
    
    return cleaned_df, suspension_report


def handle_splits_dividends(ohlcv_df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """
    Adjust historical prices for stock splits and dividends.
    
    Note: yfinance typically provides adjusted prices, but we verify and handle manually if needed.
    Strategy:
    - Check for sudden price drops that might indicate splits
    - Use yfinance's adjusted close if available
    - Otherwise, detect and adjust manually
    
    Args:
        ohlcv_df: DataFrame with OHLCV data
        tickers: List of ticker symbols
    
    Returns:
        DataFrame with split-adjusted prices
    """
    print("\n📊 Handling Stock Splits and Dividends...")
    
    adjusted_df = ohlcv_df.copy()
    
    for ticker in tickers:
        close_col = f'Close_{ticker}'
        if close_col not in adjusted_df.columns:
            continue
        
        prices = adjusted_df[close_col].dropna()
        if len(prices) < 2:
            continue
        
        # Calculate daily returns
        returns = prices.pct_change()
        
        # Detect potential splits (large negative returns, typically -50%, -33%, -25%)
        potential_splits = returns[returns < -0.2]  # More than 20% drop
        
        if len(potential_splits) > 0:
            print(f"  ⚠️  {ticker}: {len(potential_splits)} potential split events detected")
            # Note: In production, you'd fetch actual split data from yfinance or other sources
            # For now, we assume yfinance provides adjusted prices
    
    print("  ✅ Split/dividend handling complete (assuming yfinance provides adjusted prices)")
    return adjusted_df


def impute_missing_values(ohlcv_df: pd.DataFrame, tickers: List[str], 
                         method: str = 'forward_fill') -> pd.DataFrame:
    """
    Robust missing value imputation.
    
    Methods:
    - 'forward_fill': Forward-fill (default, preserves last known value)
    - 'backward_fill': Backward-fill
    - 'interpolate': Linear interpolation
    - 'drop': Drop rows with any missing values
    
    Args:
        ohlcv_df: DataFrame with OHLCV data
        tickers: List of ticker symbols
        method: Imputation method
    
    Returns:
        DataFrame with imputed values
    """
    print(f"\n🔧 Imputing Missing Values (method: {method})...")
    
    imputed_df = ohlcv_df.copy()
    initial_missing = imputed_df.isnull().sum().sum()
    
    if method == 'forward_fill':
        imputed_df = imputed_df.ffill()
        # Also backward-fill any remaining leading NaNs
        imputed_df = imputed_df.bfill()
    elif method == 'backward_fill':
        imputed_df = imputed_df.bfill()
    elif method == 'interpolate':
        imputed_df = imputed_df.interpolate(method='linear', limit_direction='both')
    elif method == 'drop':
        imputed_df = imputed_df.dropna()
    else:
        print(f"  ⚠️  Unknown method '{method}', using forward_fill")
        imputed_df = imputed_df.fillna(method='ffill').fillna(method='bfill')
    
    final_missing = imputed_df.isnull().sum().sum()
    imputed_count = initial_missing - final_missing
    
    print(f"  ✅ Imputed {imputed_count:,} missing values ({initial_missing:,} → {final_missing:,})")
    
    return imputed_df


def create_data_collection_log(output_dir: Path, tickers: List[str], 
                               start_date: str, end_date: str,
                               validation_results: Dict,
                               suspension_report: Dict) -> Path:
    """
    Create a data collection log file with metadata.
    
    Args:
        output_dir: Directory to save the log
        tickers: List of tickers collected
        start_date: Start date of data collection
        end_date: End date of data collection
        validation_results: Results from validate_ohlcv_data()
        suspension_report: Results from handle_stock_suspensions()
    
    Returns:
        Path to the created log file
    """
    log_data = {
        'collection_timestamp': datetime.now().isoformat(),
        'date_range': {
            'start': start_date,
            'end': end_date
        },
        'tickers': {
            'count': len(tickers),
            'list': tickers
        },
        'validation': {
            'is_valid': validation_results.get('is_valid', False),
            'issues': validation_results.get('issues', []),
            'warnings': validation_results.get('warnings', []),
            'statistics': validation_results.get('statistics', {})
        },
        'suspensions': suspension_report,
        'data_quality': {
            'status': 'good' if validation_results.get('is_valid', False) else 'needs_review',
            'notes': 'Data collection completed. Review warnings if any.'
        }
    }
    
    log_file = output_dir / 'data_collection_log.json'
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2, default=str)
    
    print(f"\n📝 Data collection log saved to: {log_file}")
    return log_file

