# phase1_feature_engineering.py
import pandas as pd
import numpy as np
import talib
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
# Set up paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_EDGES_DIR = PROJECT_ROOT / "data" / "edges"

# Ensure output directories exist
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
DATA_EDGES_DIR.mkdir(parents=True, exist_ok=True)

# --- Utility Functions ---

def load_raw_data():
    """Load raw data files from the 'data/raw' directory."""
    print("📁 Loading raw data...")
    
    # Load OHLCV data (saved as flat columns, e.g., 'Close_AAPL')
    ohlcv_df = pd.read_csv(DATA_RAW_DIR / 'stock_prices_ohlcv_raw.csv', index_col='Date', parse_dates=True)
    
    # Load fundamental data (simulated as quarterly/annual)
    fund_df = pd.read_csv(DATA_RAW_DIR / 'fundamentals_raw.csv', index_col='Date', parse_dates=True)
    
    # Load sentiment/macro data
    sentiment_macro_df = pd.read_csv(DATA_RAW_DIR / 'sentiment_macro_raw.csv', index_col='Date', parse_dates=True)
    
    # --- Identify Tickers ---
    # Assuming columns follow the pattern 'Feature_TICKER'
    tickers = sorted(list(set(col.split('_')[-1] for col in ohlcv_df.columns if '_' in col)))
    
    print(f"✅ Loaded raw data for {len(tickers)} tickers.")
    return ohlcv_df, fund_df, sentiment_macro_df, tickers

def align_data(ohlcv_df, fund_df, sentiment_macro_df):
    """Aligns all time series data to the daily trading calendar."""
    print("⏳ Aligning and imputing data...")
    
    # Define the master date index (all trading days from OHLCV)
    date_index = ohlcv_df.index
    
    # Forward-fill fundamentals (sparse data)
    fund_aligned = fund_df.reindex(date_index).ffill()
    
    # Forward-fill sentiment/macro data
    sentiment_macro_aligned = sentiment_macro_df.reindex(date_index).ffill()
    
    # Merge and fill any remaining non-price NaNs with zero (or median)
    aligned_data = ohlcv_df.join(fund_aligned).join(sentiment_macro_aligned)
    aligned_data = aligned_data.fillna(0) # Simple imputation for demonstration
    
    return aligned_data

# --- Feature Calculation Functions ---

def calculate_technical_indicators(aligned_data, tickers):
    """
    Calculate technical indicators for each stock and return a wide DataFrame.
    
    Returns: DataFrame indexed by Date, with columns: 'Feature_TICKER'.
    """
    print("🔧 Calculating technical indicators...")
    
    technical_features = pd.DataFrame(index=aligned_data.index)
    
    for ticker in tickers:
        try:
            # Extract OHLCV data using the flat column names (e.g., 'Close_AAPL')
            open_prices = aligned_data[f'Open_{ticker}'].dropna()
            high_prices = aligned_data[f'High_{ticker}'].dropna()
            low_prices = aligned_data[f'Low_{ticker}'].dropna()
            close_prices = aligned_data[f'Close_{ticker}'].dropna()
            volume = aligned_data[f'Volume_{ticker}'].dropna()
            
            # --- Returns and Volatility ---
            # 1-, 5-, 20-day returns
            for w in [1, 5, 20]:
                log_returns = np.log(close_prices / close_prices.shift(w))
                technical_features[f'LogRet_{w}d_{ticker}'] = log_returns
            
            # Volatility (30-day Annualized)
            daily_returns = np.log(close_prices / close_prices.shift(1))
            volatility_30d = daily_returns.rolling(window=30).std() * np.sqrt(252)
            technical_features[f'Vol_30d_{ticker}'] = volatility_30d
            
            # --- TA-Lib Indicators ---
            
            # RSI (14 day)
            technical_features[f'RSI_14_{ticker}'] = talib.RSI(close_prices.values, timeperiod=14)
            
            # MACD (MACD, MACD Signal, MACD Hist)
            macd, macd_signal, macd_hist = talib.MACD(close_prices.values)
            technical_features[f'MACD_{ticker}'] = macd
            technical_features[f'MACD_Sig_{ticker}'] = macd_signal
            technical_features[f'MACD_Hist_{ticker}'] = macd_hist
            
            # Bollinger Bands Width (Normalized: (Upper-Lower) / Middle)
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices.values)
            bb_width = (bb_upper - bb_lower) / bb_middle
            technical_features[f'BB_Width_{ticker}'] = bb_width
            
            # Moving Averages Ratio (MA50 relative to price)
            sma_50 = talib.SMA(close_prices.values, timeperiod=50)
            technical_features[f'Price_to_SMA50_{ticker}'] = close_prices.values / sma_50 - 1.0

            # ATR (14 day)
            technical_features[f'ATR_14_{ticker}'] = talib.ATR(high_prices.values, low_prices.values, close_prices.values, timeperiod=14)
            
        except Exception as e:
            print(f"❌ Error calculating technical indicators for {ticker}: {e}. Skipping.")

    # Drop initial NaNs caused by indicator lookback windows (e.g., 50 days)
    technical_features = technical_features.dropna(how='all') 
    print(f"✅ Technical features calculated. Shape: {technical_features.shape}")
    return technical_features


def normalize_fundamentals_and_sentiment(aligned_data, tickers):
    """
    Standardizes fundamental, sentiment, and macro features.
    
    This function separates the fundamental/sentiment columns, normalizes them,
    and returns them ready for consolidation.
    """
    print("\n📊 Normalizing fundamental/sentiment features...")
    
    # 1. Identify Fundamental, Sentiment, and Macro columns
    # We use the raw column names from the aligned data, excluding OHLCV
    
    # Identify non-OHLCV columns for processing
    all_features_cols = [col for col in aligned_data.columns if not any(c in col for c in ['Open', 'High', 'Low', 'Close', 'Volume'])]
    
    if not all_features_cols:
        print("❌ No fundamental or sentiment columns found for normalization.")
        return pd.DataFrame(index=aligned_data.index)

    df_to_scale = aligned_data[all_features_cols].copy()
    
    # 2. Imputation and Transformation
    # Log transform is often applied to variables like P/E and Market Cap
    for col in df_to_scale.columns:
        if any(keyword in col for keyword in ['PE', 'MarketCap']) and df_to_scale[col].max() > 0:
            # Apply log transformation (add 1 to handle potential zero/negative values)
            df_to_scale[f'{col}_Log'] = np.log(df_to_scale[col].abs() + 1)
            
    # 3. Standardization (Z-score normalization)
    scaler = StandardScaler()
    # Fill remaining NaNs (should be minimal after alignment/ffill) with 0
    df_to_scale = df_to_scale.fillna(0) 
    
    # Apply scaler to all derived numerical features
    scaled_data = scaler.fit_transform(df_to_scale)
    normalized_features = pd.DataFrame(scaled_data, index=df_to_scale.index, columns=df_to_scale.columns)
    
    print(f"✅ Normalized fundamental/sentiment features. Shape: {normalized_features.shape}")
    return normalized_features.dropna()


def calculate_dynamic_edge_params(aligned_data, tickers):
    """
    Calculates the two dynamic edge parameters: Rolling Correlation and Fundamental Similarity.
    """
    print("\n🔗 Calculating Dynamic Edge Parameters...")
    
    # --- 3.1 Rolling Correlation (rho_ij,t) ---
    print("  - Calculating Rolling Correlation Matrix...")
    CORR_WINDOW = 30
    
    log_returns_daily = pd.DataFrame(index=aligned_data.index)
    for ticker in tickers:
        log_returns_daily[ticker] = np.log(aligned_data[f'Close_{ticker}'] / aligned_data[f'Close_{ticker}'].shift(1))

    # Function to compute correlation matrix for a rolling window
    def compute_rolling_corr(df):
        return df.corr()

    # Apply rolling correlation (requires significant computation)
    # The result is a Multi-Indexed Series that we save as pickle for efficient storage
    rolling_corr_series = log_returns_daily.rolling(window=CORR_WINDOW).apply(
        lambda x: compute_rolling_corr(x), raw=False
    ).stack().rename('correlation')
    
    # Remove self-correlation and NaN entries (first 30 days)
    rolling_corr_series = rolling_corr_series[rolling_corr_series.index.get_level_values(1) != rolling_corr_series.index.get_level_values(2)].dropna()

    corr_path = DATA_EDGES_DIR / 'edges_dynamic_corr_params.pkl'
    rolling_corr_series.to_pickle(corr_path)
    print(f"✅ Rolling correlation parameters saved to: {corr_path}")


    # --- 3.2 Fundamental Similarity (s_ij) ---
    # This is calculated once based on the latest available normalized metrics
    print("  - Calculating Fundamental Similarity Matrix (latest data)...")
    
    # Get all normalized fundamental features from the processed file
    # We use the features generated in the normalization step
    
    # Get the latest fundamental feature vector for each stock
    # We assume 'PE' and 'ROE' are the key metrics for similarity
    fund_cols = [col for col in aligned_data.columns if any(metric in col for metric in ['PE', 'ROE'])]
    latest_date = aligned_data.index[-1]
    
    # Reformat data to be (N_stocks x N_metrics) for the latest date
    metric_vectors = {}
    for ticker in tickers:
        # Extract the fundamental metrics for the specific ticker
        vector = []
        for col in fund_cols:
            if ticker in col:
                 # Need to find the correct normalized vector for the latest date
                 # For simplicity, we use the raw aligned data's latest vector here (will be normalized in normalize_fundamentals_and_sentiment)
                 vector.append(aligned_data.loc[latest_date, col])
        
        if vector:
            metric_vectors[ticker] = vector
    
    ticker_list = list(metric_vectors.keys())
    # Note: We should ideally use the *normalized* features from `normalize_fundamentals_and_sentiment` here
    # Since we don't return them from that function, we use the simple raw vector for demonstration.
    
    if ticker_list:
        sim_matrix = pd.DataFrame(index=ticker_list, columns=ticker_list)
        for i in ticker_list:
            for j in ticker_list:
                # Cosine Similarity between the latest fundamental feature vectors
                sim = 1 - cosine(metric_vectors[i], metric_vectors[j])
                sim_matrix.loc[i, j] = sim
        
        fund_sim_path = DATA_EDGES_DIR / 'edges_dynamic_fund_sim_params.csv'
        sim_matrix.to_csv(fund_sim_path)
        print(f"✅ Fundamental Similarity matrix (latest) saved to: {fund_sim_path}")


def consolidate_node_features(technical_df, normalized_df, tickers):
    """
    Consolidate all feature types (Technical, Fundamental, Sentiment/Macro) 
    into a single, wide, time-aligned DataFrame (X_t).
    
    Args:
        technical_df (pd.DataFrame): Date-indexed technical features (wide format)
        normalized_df (pd.DataFrame): Date-indexed fundamental/sentiment features (wide format)
    
    Returns:
        pd.DataFrame: Final X_t matrix.
    """
    print("\n⭐ Consolidating Final Node Features (X_t)...")
    
    # Combine the two wide feature sets based on their Date index
    final_X_t = technical_df.join(normalized_df, how='inner')
    
    # Final cleanup: drop rows with any NaNs that might remain
    final_X_t = final_X_t.dropna()
    
    # Save the final X_t matrix
    output_file = DATA_PROCESSED_DIR / "node_features_X_t_final.csv"
    final_X_t.to_csv(output_file)
    
    print(f"✅ Final X_t matrix saved to: {output_file}")
    print(f"Final X_t Shape: {final_X_t.shape}")
    
    return final_X_t

# --- Main Execution ---

def main():
    """Main function to execute the feature engineering pipeline."""
    print("🚀 Starting Phase 1: Feature Engineering")
    print("=" * 50)
    
    try:
        # 0. Load and Align Data
        ohlcv_df, fund_df, sentiment_macro_df, tickers = load_raw_data()
        aligned_data = align_data(ohlcv_df, fund_df, sentiment_macro_df)
        
        # 1. Calculate Technical Indicators
        technical_features = calculate_technical_indicators(aligned_data, tickers)
        
        # 2. Normalize Fundamental, Sentiment, and Macro Features
        normalized_features = normalize_fundamentals_and_sentiment(aligned_data, tickers)
        
        # 3. Calculate Dynamic Edge Parameters
        calculate_dynamic_edge_params(aligned_data, tickers)
        
        # 4. Consolidate All Features (X_t)
        if technical_features is not None:
            consolidate_node_features(technical_features, normalized_features, tickers)
        
        print("\n" + "=" * 50)
        print("✅ Phase 1: Feature Engineering Complete!")
        print(f"📁 Processed data in: {DATA_PROCESSED_DIR} and {DATA_EDGES_DIR}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Raw data file missing. Please run phase1_data_collection.py first.")
        print(f"Missing file detail: {e}")
    except Exception as e:
        print(f"❌ Error in feature engineering pipeline: {e}")

if __name__ == "__main__":
    # Dependencies: pip install pandas numpy ta-lib scikit-learn scipy
    main()