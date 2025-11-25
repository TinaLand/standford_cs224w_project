"""
Test Data Validation on Existing Data
======================================

This script tests the new data validation and cleaning functions
on existing data files without re-downloading.

Usage:
    python scripts/test_data_validation.py
"""

import pandas as pd
from pathlib import Path
import sys

# Add scripts directory to path
sys.path.append(str(Path(__file__).resolve().parent))

from utils_data_validation import (
    validate_ohlcv_data,
    align_trading_calendars,
    handle_stock_suspensions,
    handle_splits_dividends,
    impute_missing_values,
    create_data_collection_log
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
OHLCV_FILE = DATA_RAW_DIR / "stock_prices_ohlcv_raw.csv"


def main():
    print("=" * 60)
    print("Testing Data Validation on Existing Data")
    print("=" * 60)
    
    # 1. Load existing data
    print(f"\n📂 Loading existing OHLCV data from: {OHLCV_FILE}")
    
    if not OHLCV_FILE.exists():
        print(f"❌ Error: Data file not found at {OHLCV_FILE}")
        print("   Please run Phase 1 data collection first.")
        return
    
    try:
        ohlcv_df = pd.read_csv(OHLCV_FILE, index_col=0, parse_dates=True)
        print(f"✅ Loaded data: {ohlcv_df.shape}")
        print(f"   Date range: {ohlcv_df.index.min().date()} to {ohlcv_df.index.max().date()}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # 2. Extract ticker list
    tickers = sorted(list(set(col.split('_')[-1] for col in ohlcv_df.columns if '_' in col)))
    print(f"\n📊 Found {len(tickers)} tickers: {', '.join(tickers[:10])}...")
    
    # 3. Run validation
    print("\n" + "=" * 60)
    print("STEP 1: Data Quality Validation")
    print("=" * 60)
    validation_results = validate_ohlcv_data(ohlcv_df, tickers)
    
    # 4. Align trading calendars
    print("\n" + "=" * 60)
    print("STEP 2: Trading Calendar Alignment")
    print("=" * 60)
    aligned_df = align_trading_calendars(ohlcv_df.copy(), tickers)
    
    # 5. Handle suspensions
    print("\n" + "=" * 60)
    print("STEP 3: Stock Suspension Handling")
    print("=" * 60)
    cleaned_df, suspension_report = handle_stock_suspensions(aligned_df, tickers)
    
    # 6. Handle splits/dividends
    print("\n" + "=" * 60)
    print("STEP 4: Split/Dividend Handling")
    print("=" * 60)
    split_adjusted_df = handle_splits_dividends(cleaned_df, tickers)
    
    # 7. Impute missing values
    print("\n" + "=" * 60)
    print("STEP 5: Missing Value Imputation")
    print("=" * 60)
    final_df = impute_missing_values(split_adjusted_df, tickers, method='forward_fill')
    
    # 8. Create data collection log
    print("\n" + "=" * 60)
    print("STEP 6: Creating Data Collection Log")
    print("=" * 60)
    log_file = create_data_collection_log(
        DATA_RAW_DIR,
        tickers,
        str(ohlcv_df.index.min().date()),
        str(ohlcv_df.index.max().date()),
        validation_results,
        suspension_report
    )
    
    # 9. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Original data shape: {ohlcv_df.shape}")
    print(f"Final data shape: {final_df.shape}")
    print(f"Missing values (original): {ohlcv_df.isnull().sum().sum():,}")
    print(f"Missing values (final): {final_df.isnull().sum().sum():,}")
    print(f"\nValidation status: {'✅ PASSED' if validation_results['is_valid'] else '❌ FAILED'}")
    print(f"Warnings: {len(validation_results.get('warnings', []))}")
    print(f"Suspensions detected: {len(suspension_report)} tickers")
    print(f"\nData collection log: {log_file}")
    
    # 10. Optionally save cleaned data
    save_cleaned = input("\n💾 Save cleaned data to 'stock_prices_ohlcv_cleaned.csv'? (y/n): ").lower().strip()
    if save_cleaned == 'y':
        cleaned_file = DATA_RAW_DIR / "stock_prices_ohlcv_cleaned.csv"
        final_df.to_csv(cleaned_file, index_label='Date')
        print(f"✅ Cleaned data saved to: {cleaned_file}")
    
    print("\n✅ Data validation test complete!")


if __name__ == '__main__':
    main()

