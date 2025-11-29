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
# NOTE: PROJECT_ROOT should point to the project root (one level above `src/`)
# `__file__` is in `src/data/`, so we need `.parent.parent.parent`
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_EDGES_DIR = PROJECT_ROOT / "data" / "edges"

# Ensure output directories exist
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
DATA_EDGES_DIR.mkdir(parents=True, exist_ok=True)

# --- Utility Functions ---

def _read_time_series(path):
    df = pd.read_csv(path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
    else:
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df = df.set_index(df.columns[0])
        df.index.name = 'Date'
    return df


def load_raw_data():
    """Load raw data files from the 'data/raw' directory."""
    print("📁 Loading raw data...")
    
    # Load OHLCV data (saved as flat columns, e.g., 'Close_AAPL')
    ohlcv_path = DATA_RAW_DIR / 'stock_prices_ohlcv_raw.csv'
    ohlcv_df = _read_time_series(ohlcv_path)
    
    # Load fundamental data (simulated as quarterly/annual)
    fund_df = _read_time_series(DATA_RAW_DIR / 'fundamentals_raw.csv')
    
    # Load sentiment/macro data
    sentiment_macro_df = _read_time_series(DATA_RAW_DIR / 'sentiment_macro_raw.csv')
    
    # --- Identify Tickers ---
    # Assuming columns follow the pattern 'Feature_TICKER'
    tickers = sorted(list(set(col.split('_')[-1] for col in ohlcv_df.columns if '_' in col)))
    
    if len(tickers) < 2 or ohlcv_df.empty:
        print("⚠️  Warning: OHLCV raw data is empty. Generating synthetic price series for fallback.")
        fallback_tickers = [
            'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'BRK-B', 'LLY', 'TSLA', 'V',
            'JPM', 'XOM', 'JNJ', 'WMT', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV',
            'KO', 'PEP', 'AVGO', 'COST', 'PFE', 'ADBE', 'CSCO', 'CMCSA', 'NFLX', 'DIS',
            'ACN', 'CRM', 'TMO', 'QCOM', 'TXN', 'UNH', 'BAC', 'MCD', 'ORCL', 'INTC',
            'SBUX', 'CAT', 'GE', 'NKE', 'AXP', 'IBM', 'MMM', 'VZ', 'FDX', 'GOOG'
        ]
        date_index = pd.date_range(start='2018-01-01', end='2024-12-31', freq='B')
        synthetic_data = {}
        rng = np.random.default_rng(seed=42)
        for ticker in fallback_tickers:
            price = 100 * np.exp(np.cumsum(rng.normal(loc=0.0003, scale=0.02, size=len(date_index))))
            open_price = price * (1 + rng.normal(loc=0, scale=0.002, size=len(date_index)))
            high_price = np.maximum(open_price, price) * (1 + rng.uniform(0, 0.01, size=len(date_index)))
            low_price = np.minimum(open_price, price) * (1 - rng.uniform(0, 0.01, size=len(date_index)))
            volume = rng.integers(low=1_000_000, high=10_000_000, size=len(date_index))
            
            synthetic_data[f'Open_{ticker}'] = open_price
            synthetic_data[f'High_{ticker}'] = high_price
            synthetic_data[f'Low_{ticker}'] = low_price
            synthetic_data[f'Close_{ticker}'] = price
            synthetic_data[f'Volume_{ticker}'] = volume
        
        ohlcv_df = pd.DataFrame(synthetic_data, index=date_index)
        ohlcv_df.index.name = 'Date'
        ohlcv_df.to_csv(ohlcv_path)
        tickers = fallback_tickers
        print(f"✅ Synthetic OHLCV data generated with shape: {ohlcv_df.shape}")
    
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
            # Keep as pandas Series for shift/rolling operations, convert to float64 only for TA-Lib
            open_prices = aligned_data[f'Open_{ticker}'].dropna()
            high_prices = aligned_data[f'High_{ticker}'].dropna()
            low_prices = aligned_data[f'Low_{ticker}'].dropna()
            close_prices = aligned_data[f'Close_{ticker}'].dropna()
            volume = aligned_data[f'Volume_{ticker}'].dropna()
            
            # --- Returns and Volatility ---
            # Note: close_prices is still a pandas Series (for shift/rolling operations)
            # We'll convert to float64 only when passing to TA-Lib functions
            
            # 1-, 5-, 20-day returns
            for w in [1, 5, 20]:
                # Ensure result is pandas Series (not numpy array)
                log_returns = pd.Series(
                    np.log(close_prices / close_prices.shift(w)),
                    index=close_prices.index
                )
                technical_features[f'LogRet_{w}d_{ticker}'] = log_returns
            
            # Volatility (30-day Annualized)
            # Ensure daily_returns is pandas Series for rolling operations
            daily_returns = pd.Series(
                np.log(close_prices / close_prices.shift(1)),
                index=close_prices.index
            )
            volatility_30d = daily_returns.rolling(window=30).std() * np.sqrt(252)
            technical_features[f'Vol_30d_{ticker}'] = volatility_30d
            
            # --- TA-Lib Indicators ---
            
            # RSI (14 day)
            # Ensure values are float64 for TA-Lib
            technical_features[f'RSI_14_{ticker}'] = talib.RSI(close_prices.values.astype(np.float64), timeperiod=14)
            
            # MACD (MACD, MACD Signal, MACD Hist)
            # Ensure float64 for TA-Lib
            macd, macd_signal, macd_hist = talib.MACD(close_prices.values.astype(np.float64))
            technical_features[f'MACD_{ticker}'] = macd
            technical_features[f'MACD_Sig_{ticker}'] = macd_signal
            technical_features[f'MACD_Hist_{ticker}'] = macd_hist
            
            # Bollinger Bands Width (Normalized: (Upper-Lower) / Middle)
            # Ensure float64 for TA-Lib
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices.values.astype(np.float64))
            bb_width = (bb_upper - bb_lower) / bb_middle
            technical_features[f'BB_Width_{ticker}'] = bb_width
            
            # Moving Averages Ratio (MA50 relative to price)
            # Ensure float64 for TA-Lib
            sma_50 = talib.SMA(close_prices.values.astype(np.float64), timeperiod=50)
            technical_features[f'Price_to_SMA50_{ticker}'] = close_prices.values / (sma_50 + 1e-8) - 1.0

            # ATR (14 day)
            technical_features[f'ATR_14_{ticker}'] = talib.ATR(
                high_prices.values.astype(np.float64),
                low_prices.values.astype(np.float64),
                close_prices.values.astype(np.float64),
                timeperiod=14
            )
            
            # --- Additional Technical Indicators (Enhanced Feature Engineering) ---
            
            # Stochastic Oscillator (%K, %D)
            # Ensure float64 for TA-Lib
            slowk, slowd = talib.STOCH(
                high_prices.values.astype(np.float64),
                low_prices.values.astype(np.float64),
                close_prices.values.astype(np.float64),
                fastk_period=14, slowk_period=3, slowd_period=3
            )
            technical_features[f'STOCH_K_{ticker}'] = slowk
            technical_features[f'STOCH_D_{ticker}'] = slowd
            
            # ADX (Average Directional Index) - Trend strength indicator
            technical_features[f'ADX_14_{ticker}'] = talib.ADX(
                high_prices.values.astype(np.float64),
                low_prices.values.astype(np.float64),
                close_prices.values.astype(np.float64),
                timeperiod=14
            )
            
            # CCI (Commodity Channel Index)
            technical_features[f'CCI_14_{ticker}'] = talib.CCI(
                high_prices.values.astype(np.float64),
                low_prices.values.astype(np.float64),
                close_prices.values.astype(np.float64),
                timeperiod=14
            )
            
            # Williams %R
            technical_features[f'WILLR_14_{ticker}'] = talib.WILLR(
                high_prices.values.astype(np.float64),
                low_prices.values.astype(np.float64),
                close_prices.values.astype(np.float64),
                timeperiod=14
            )
            
            # OBV (On Balance Volume) - Volume-based momentum
            obv = talib.OBV(close_prices.values.astype(np.float64), volume.values.astype(np.float64))
            # Normalize OBV by price to make it comparable across stocks
            technical_features[f'OBV_Norm_{ticker}'] = obv / (close_prices.values + 1e-8)
            
            # AD (Accumulation/Distribution Line)
            ad = talib.AD(
                high_prices.values.astype(np.float64),
                low_prices.values.astype(np.float64),
                close_prices.values.astype(np.float64),
                volume.values.astype(np.float64)
            )
            # Normalize AD by price
            technical_features[f'AD_Norm_{ticker}'] = ad / (close_prices.values + 1e-8)
            
            # Momentum (10-day)
            technical_features[f'MOM_10_{ticker}'] = talib.MOM(close_prices.values.astype(np.float64), timeperiod=10)
            
            # Rate of Change (ROC, 10-day)
            technical_features[f'ROC_10_{ticker}'] = talib.ROC(close_prices.values.astype(np.float64), timeperiod=10)
            
            # Additional Moving Averages and Crossovers
            sma_20 = talib.SMA(close_prices.values.astype(np.float64), timeperiod=20)
            sma_200 = talib.SMA(close_prices.values.astype(np.float64), timeperiod=200)
            ema_12 = talib.EMA(close_prices.values.astype(np.float64), timeperiod=12)
            ema_26 = talib.EMA(close_prices.values.astype(np.float64), timeperiod=26)
            
            # MA Crossovers (price relative to MAs)
            technical_features[f'Price_to_SMA20_{ticker}'] = close_prices.values / (sma_20 + 1e-8) - 1.0
            technical_features[f'Price_to_SMA200_{ticker}'] = close_prices.values / (sma_200 + 1e-8) - 1.0
            technical_features[f'SMA20_to_SMA50_{ticker}'] = sma_20 / (sma_50 + 1e-8) - 1.0
            technical_features[f'EMA12_to_EMA26_{ticker}'] = ema_12 / (ema_26 + 1e-8) - 1.0
            
            # Volume-based indicators
            # Volume Rate of Change (10-day)
            volume_roc = talib.ROC(volume.values.astype(np.float64), timeperiod=10)
            technical_features[f'Volume_ROC_10_{ticker}'] = volume_roc
            
            # Volume Moving Average Ratio
            volume_ma = talib.SMA(volume.values.astype(np.float64), timeperiod=20)
            technical_features[f'Volume_to_MA20_{ticker}'] = volume.values / (volume_ma + 1e-8) - 1.0
            
            # Price-Volume Trend (PVT)
            pvt = talib.OBV(close_prices.values.astype(np.float64), volume.values.astype(np.float64))
            # Convert pvt back to pandas Series for rolling operations
            pvt_series = pd.Series(pvt, index=close_prices.index)
            pvt_normalized = (pvt_series - pvt_series.rolling(window=20).mean()) / (pvt_series.rolling(window=20).std() + 1e-8)
            technical_features[f'PVT_Norm_{ticker}'] = pvt_normalized.values
            
            # Additional volatility measures
            # Historical Volatility (20-day)
            hv_20d = daily_returns.rolling(window=20).std() * np.sqrt(252)
            technical_features[f'HV_20d_{ticker}'] = hv_20d
            
            # True Range normalized by price
            tr = talib.TRANGE(
                high_prices.values.astype(np.float64),
                low_prices.values.astype(np.float64),
                close_prices.values.astype(np.float64)
            )
            technical_features[f'TR_Norm_{ticker}'] = tr / (close_prices.values + 1e-8)
            
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

    # --- FIX: Use optimized Pandas rolling correlation ---
    print(f"    Computing rolling cross-correlations over a {CORR_WINDOW}-day window...")
    
    # 1. Calculate the rolling correlation matrix (MultiIndex Output)
    # The result is a 3D structure (Time x Stock1 x Stock2) flattened to a MultiIndex Series
    rolling_corr_matrix = log_returns_daily.rolling(window=CORR_WINDOW).corr()
    
    # 2. Extract and format the correlation series
    # The output MultiIndex is (Date, Stock1, Stock2). We need to filter it.
    
    # Reset index to convert MultiIndex to columns (Date, Stock1, Stock2)
    rolling_corr_df = rolling_corr_matrix.stack().rename('correlation').to_frame().reset_index()
    rolling_corr_df.columns = ['Date', 'ticker1', 'ticker2', 'correlation']
    
    # Remove self-correlation (ticker1 == ticker2) and NaNs (from the first CORR_WINDOW days)
    rolling_corr_series = rolling_corr_df[rolling_corr_df['ticker1'] < rolling_corr_df['ticker2']]
    rolling_corr_series = rolling_corr_series.set_index(['Date', 'ticker1', 'ticker2'])['correlation'].dropna()
    
    if len(rolling_corr_series) > 0:
        corr_path = DATA_EDGES_DIR / 'edges_dynamic_corr_params.pkl'
        rolling_corr_series.to_pickle(corr_path)
        print(f"✅ Rolling correlation parameters saved to: {corr_path}")
        print(f"    Total correlation pairs computed: {len(rolling_corr_series)} (time-series pairs)")
    else:
        print("⚠️ No correlation data computed - insufficient data quality after rolling window.")


    # --- 3.2 Fundamental Similarity (s_ij) ---
    print("  - Calculating Fundamental Similarity Matrix (latest data)...")
    
    # 1. Define Universal Fundamental Metrics
    # Find ALL unique fundamental/ROE/PE suffixes across the aligned data.
    # Columns look like 'MSFT_PE', 'AAPL_ROE', etc.
    # We want metric suffixes like 'PE', 'ROE', 'PE_Log', 'ROE_Log'.
    universal_metrics = sorted(list(set(
        '_'.join(col.split('_')[1:])
        for col in aligned_data.columns
        if any(metric in col for metric in ['PE', 'ROE'])  # Focus on key metrics
    )))
    
    print(f"    Universal fundamental metrics identified: {universal_metrics}")
    
    # Reformat data to be (N_stocks x N_metrics) for the latest date
    latest_date = aligned_data.index[-1]
    metric_vectors = {}
    
    for ticker in tickers:
        vector = []
        is_valid = False
        
        # Build the vector using the universal list of metrics
        for metric_suffix in universal_metrics:
            # The column name is reconstructed: e.g., 'AAPL_PE', 'AAPL_ROE'
            col_name = f'{ticker}_{metric_suffix}'
            
            if col_name in aligned_data.columns:
                vector.append(aligned_data.loc[latest_date, col_name])
                is_valid = True
            else:
                # IMPORTANT FIX: Append 0 for missing metrics, ensuring uniform vector length
                vector.append(0.0)
        
        if is_valid:
            metric_vectors[ticker] = vector
    
    ticker_list = list(metric_vectors.keys())
    
    if ticker_list:
        sim_matrix = pd.DataFrame(index=ticker_list, columns=ticker_list)
        for i in ticker_list:
            for j in ticker_list:
                if i == j:
                    sim_matrix.loc[i, j] = 1.0
                else:
                    try:
                        # Cosine Similarity between the latest fundamental feature vectors
                        v_i = metric_vectors[i]
                        v_j = metric_vectors[j]
                        
                        # NOTE: Since we ensured len(v_i) == len(v_j) using universal_metrics, this should pass.
                        sim = 1 - cosine(v_i, v_j)
                        sim_matrix.loc[i, j] = sim
                    except Exception as e:
                        sim_matrix.loc[i, j] = 0.0 # Safety fallback
        
        # Save a dense similarity matrix (tickers x tickers) for analysis.
        # NOTE: Edge list for GNN graph construction is computed separately
        # in `src/data/edge_parameters.py` and saved to
        # 'edges_dynamic_fund_sim_params.csv'.
        fund_sim_path = DATA_EDGES_DIR / 'edges_fundamental_similarity_matrix.csv'
        sim_matrix.to_csv(fund_sim_path)
        print(f"✅ Fundamental Similarity matrix (latest) saved to: {fund_sim_path}")
        print(f"    Similarity matrix shape: {sim_matrix.shape}")
    else:
        print("⚠️ Fundamental Similarity: No valid metric vectors found.")


def calculate_static_edge_data(tickers):
    """
    Calculate static edge relationships: Sector, Industry, and Market Cap-based connections.
    
    This creates time-invariant edge connections based on:
    1. Sector relationships (same sector = strong connection)
    2. Industry relationships (same industry = moderate connection)  
    3. Market cap tiers (similar size = weak connection)
    
    Returns:
        pd.DataFrame: Static edge connections with weights
    """
    print("\n🏢 Calculating Static Edge Data (Sector, Industry, Market Cap)...")
    
    # Define sector and industry mappings for major stocks
    # In a real implementation, this would come from financial APIs
    stock_metadata = {
        'AAPL': {'sector': 'Technology', 'industry': 'Consumer Electronics', 'market_cap_tier': 'Mega'},
        'MSFT': {'sector': 'Technology', 'industry': 'Software', 'market_cap_tier': 'Mega'},
        'GOOGL': {'sector': 'Technology', 'industry': 'Internet Services', 'market_cap_tier': 'Mega'},
        'GOOG': {'sector': 'Technology', 'industry': 'Internet Services', 'market_cap_tier': 'Mega'},
        'AMZN': {'sector': 'Consumer Discretionary', 'industry': 'E-commerce', 'market_cap_tier': 'Mega'},
        'NVDA': {'sector': 'Technology', 'industry': 'Semiconductors', 'market_cap_tier': 'Mega'},
        'META': {'sector': 'Technology', 'industry': 'Social Media', 'market_cap_tier': 'Mega'},
        'TSLA': {'sector': 'Consumer Discretionary', 'industry': 'Electric Vehicles', 'market_cap_tier': 'Large'},
        'JPM': {'sector': 'Financial Services', 'industry': 'Banking', 'market_cap_tier': 'Mega'},
        'V': {'sector': 'Financial Services', 'industry': 'Payment Processing', 'market_cap_tier': 'Mega'},
        'JNJ': {'sector': 'Healthcare', 'industry': 'Pharmaceuticals', 'market_cap_tier': 'Mega'},
        'WMT': {'sector': 'Consumer Staples', 'industry': 'Retail', 'market_cap_tier': 'Mega'},
        'PG': {'sector': 'Consumer Staples', 'industry': 'Consumer Goods', 'market_cap_tier': 'Large'},
        'HD': {'sector': 'Consumer Discretionary', 'industry': 'Home Improvement', 'market_cap_tier': 'Large'},
        'CVX': {'sector': 'Energy', 'industry': 'Oil & Gas', 'market_cap_tier': 'Large'},
        'XOM': {'sector': 'Energy', 'industry': 'Oil & Gas', 'market_cap_tier': 'Large'},
        'BAC': {'sector': 'Financial Services', 'industry': 'Banking', 'market_cap_tier': 'Large'},
        'ABBV': {'sector': 'Healthcare', 'industry': 'Pharmaceuticals', 'market_cap_tier': 'Large'},
        'PFE': {'sector': 'Healthcare', 'industry': 'Pharmaceuticals', 'market_cap_tier': 'Large'},
        'KO': {'sector': 'Consumer Staples', 'industry': 'Beverages', 'market_cap_tier': 'Large'},
        'PEP': {'sector': 'Consumer Staples', 'industry': 'Beverages', 'market_cap_tier': 'Large'},
        'CSCO': {'sector': 'Technology', 'industry': 'Networking', 'market_cap_tier': 'Large'},
        'ADBE': {'sector': 'Technology', 'industry': 'Software', 'market_cap_tier': 'Large'},
        'NFLX': {'sector': 'Communication Services', 'industry': 'Streaming', 'market_cap_tier': 'Large'},
        'INTC': {'sector': 'Technology', 'industry': 'Semiconductors', 'market_cap_tier': 'Large'},
        'ORCL': {'sector': 'Technology', 'industry': 'Software', 'market_cap_tier': 'Large'},
    }
    
    # Add default metadata for any missing tickers
    for ticker in tickers:
        if ticker not in stock_metadata:
            stock_metadata[ticker] = {
                'sector': 'Other', 
                'industry': 'Other', 
                'market_cap_tier': 'Large'
            }
    
    static_edges = []
    
    # Calculate static relationships for all pairs
    for i, ticker1 in enumerate(tickers):
        for j, ticker2 in enumerate(tickers):
            if i < j:  # Avoid duplicates
                meta1 = stock_metadata.get(ticker1, {})
                meta2 = stock_metadata.get(ticker2, {})
                
                # Initialize edge weight
                edge_weight = 0.0
                connection_types = []
                
                # 1. Sector Connection (Strongest)
                if meta1.get('sector') == meta2.get('sector') and meta1.get('sector') != 'Other':
                    edge_weight += 0.8
                    connection_types.append('same_sector')
                
                # 2. Industry Connection (Moderate)
                if meta1.get('industry') == meta2.get('industry') and meta1.get('industry') != 'Other':
                    edge_weight += 0.6
                    connection_types.append('same_industry')
                
                # 3. Market Cap Tier Connection (Weak)
                if meta1.get('market_cap_tier') == meta2.get('market_cap_tier'):
                    edge_weight += 0.2
                    connection_types.append('same_cap_tier')
                
                # Only store edges with some connection
                if edge_weight > 0:
                    static_edges.append({
                        'ticker1': ticker1,
                        'ticker2': ticker2,
                        'static_weight': min(edge_weight, 1.0),  # Cap at 1.0
                        'connection_types': ','.join(connection_types),
                        'sector1': meta1.get('sector'),
                        'sector2': meta2.get('sector'),
                        'industry1': meta1.get('industry'),
                        'industry2': meta2.get('industry')
                    })
    
    if static_edges:
        static_df = pd.DataFrame(static_edges)
        
        # Save static edge data
        static_path = DATA_EDGES_DIR / 'edges_static_connections.csv'
        static_df.to_csv(static_path, index=False)
        
        # Statistics
        print(f"✅ Static edge connections saved to: {static_path}")
        print(f"    Total static connections: {len(static_df)}")
        print(f"    Same sector pairs: {static_df['connection_types'].str.contains('same_sector').sum()}")
        print(f"    Same industry pairs: {static_df['connection_types'].str.contains('same_industry').sum()}")
        print(f"    Same cap tier pairs: {static_df['connection_types'].str.contains('same_cap_tier').sum()}")
        
        # Show sector distribution
        sector_counts = {}
        for ticker in tickers:
            sector = stock_metadata.get(ticker, {}).get('sector', 'Other')
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        print(f"    Sector distribution: {dict(sorted(sector_counts.items(), key=lambda x: x[1], reverse=True))}")
        
        return static_df
    else:
        print("⚠️ No static connections found")
        return None


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
    final_X_t.index.name = 'Date'
    
    # Save the final X_t matrix
    output_file = DATA_PROCESSED_DIR / "node_features_X_t_final.csv"
    final_X_t.to_csv(output_file, index_label='Date')
    
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
        
        # 4. Calculate Static Edge Data (Sector, Industry, Market Cap)
        calculate_static_edge_data(tickers)
        
        # 5. Consolidate All Features (X_t)
        if technical_features is not None:
            consolidate_node_features(technical_features, normalized_features, tickers)
        
        print("\n" + "=" * 50)
        print("✅ Phase 1: Feature Engineering Complete!")
        print(f"📁 All data ready for Phase 2 Graph Construction!")
        print(f"📁 Node features: {DATA_PROCESSED_DIR}")
        print(f"📁 Edge parameters: {DATA_EDGES_DIR}")
        
        # Summary of generated files
        print(f"\n📋 Generated Files Summary:")
        print(f"   🎯 Node Features: node_features_X_t_final.csv")
        print(f"   📊 Dynamic Edges (correlation): edges_dynamic_corr_params.pkl") 
        print(f"   🔗 Fundamental Similarity (matrix for analysis): edges_fundamental_similarity_matrix.csv")
        print(f"   🏢 Static Connections: edges_static_connections.csv")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Raw data file missing. Please run phase1_data_collection.py first.")
        print(f"Missing file detail: {e}")
    except Exception as e:
        print(f"❌ Error in feature engineering pipeline: {e}")

if __name__ == "__main__":
    # Dependencies: pip install pandas numpy ta-lib scikit-learn scipy
    main()