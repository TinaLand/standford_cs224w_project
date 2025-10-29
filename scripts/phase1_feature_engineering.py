"""
Main script for node feature calculation (X_t)

Phase 1: Feature Engineering Script
==================================
This script processes raw data and generates features for nodes in the stock graph.
It creates technical indicators, normalizes fundamental data, and processes sentiment.

Key Functions:
- calculate_technical_indicators(): Computes RSI, MACD, Bollinger Bands, etc.
- normalize_fundamentals(): Standardizes fundamental metrics
- aggregate_sentiment(): Processes sentiment and macro indicators
"""

import pandas as pd
import numpy as np
import talib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set up paths
BASE_DIR = Path(__file__).parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"

def ensure_processed_directories():
    """
    Create processed data directories if they don't exist.
    """
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✅ Processed data directory ready: {DATA_PROCESSED_DIR}")

def calculate_technical_indicators(price_data):
    """
    Calculate technical indicators for each stock.
    
    Technical indicators include:
    - RSI (Relative Strength Index): Momentum oscillator (0-100)
    - MACD (Moving Average Convergence Divergence): Trend following momentum
    - Bollinger Bands: Volatility indicator
    - Moving Averages: Trend indicators
    - ATR (Average True Range): Volatility measure
    
    Args:
        price_data (pd.DataFrame): Multi-index DataFrame with OHLCV data
    
    Returns:
        pd.DataFrame: DataFrame with technical indicators for each stock
    """
    print("🔧 Calculating technical indicators...")
    
    technical_features = []
    
    # Get list of tickers from columns
    if isinstance(price_data.columns, pd.MultiIndex):
        tickers = price_data.columns.get_level_values(1).unique()
        price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    else:
        # Handle case where data might be for single ticker
        tickers = ['SINGLE_TICKER']
        price_columns = price_data.columns
    
    for ticker in tickers:
        try:
            # Extract OHLCV data for this ticker
            if isinstance(price_data.columns, pd.MultiIndex):
                open_prices = price_data[('Open', ticker)].dropna()
                high_prices = price_data[('High', ticker)].dropna()
                low_prices = price_data[('Low', ticker)].dropna()
                close_prices = price_data[('Close', ticker)].dropna()
                volume = price_data[('Volume', ticker)].dropna()
            else:
                # Assuming single ticker data
                open_prices = price_data['Open'].dropna()
                high_prices = price_data['High'].dropna()
                low_prices = price_data['Low'].dropna()
                close_prices = price_data['Close'].dropna()
                volume = price_data['Volume'].dropna()
            
            # Calculate returns
            returns = close_prices.pct_change().dropna()
            log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
            
            # Technical Indicators using TA-Lib
            # RSI (Relative Strength Index) - momentum oscillator
            rsi_14 = talib.RSI(close_prices.values, timeperiod=14)
            rsi_30 = talib.RSI(close_prices.values, timeperiod=30)
            
            # MACD (Moving Average Convergence Divergence)
            macd, macd_signal, macd_histogram = talib.MACD(close_prices.values)
            
            # Bollinger Bands - volatility indicator
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices.values)
            bb_width = (bb_upper - bb_lower) / bb_middle  # Normalized band width
            bb_position = (close_prices.values - bb_lower) / (bb_upper - bb_lower)  # Position within bands
            
            # Moving Averages
            sma_10 = talib.SMA(close_prices.values, timeperiod=10)
            sma_30 = talib.SMA(close_prices.values, timeperiod=30)
            sma_50 = talib.SMA(close_prices.values, timeperiod=50)
            ema_12 = talib.EMA(close_prices.values, timeperiod=12)
            ema_26 = talib.EMA(close_prices.values, timeperiod=26)
            
            # Price relative to moving averages
            price_to_sma10 = close_prices.values / sma_10
            price_to_sma30 = close_prices.values / sma_30
            price_to_sma50 = close_prices.values / sma_50
            
            # ATR (Average True Range) - volatility
            atr_14 = talib.ATR(high_prices.values, low_prices.values, close_prices.values, timeperiod=14)
            
            # Stochastic Oscillator
            stoch_k, stoch_d = talib.STOCH(high_prices.values, low_prices.values, close_prices.values)
            
            # Volume indicators
            volume_ma_10 = talib.SMA(volume.values, timeperiod=10)
            volume_ratio = volume.values / volume_ma_10
            
            # Volatility measures
            volatility_30d = returns.rolling(window=30).std() * np.sqrt(252)  # Annualized volatility
            
            # Create feature DataFrame for this ticker
            dates = close_prices.index
            min_length = min(len(dates), len(rsi_14), len(macd), len(bb_upper), 
                           len(sma_10), len(atr_14), len(stoch_k))
            
            # Truncate all arrays to minimum length to avoid alignment issues
            ticker_features = pd.DataFrame({
                'date': dates[-min_length:],
                'ticker': ticker,
                'close_price': close_prices.iloc[-min_length:].values,
                'returns': returns.iloc[-min_length:].values if len(returns) >= min_length else np.full(min_length, np.nan),
                'log_returns': log_returns.iloc[-min_length:].values if len(log_returns) >= min_length else np.full(min_length, np.nan),
                'volatility_30d': volatility_30d.iloc[-min_length:].values if len(volatility_30d) >= min_length else np.full(min_length, np.nan),
                
                # RSI indicators
                'rsi_14': rsi_14[-min_length:],
                'rsi_30': rsi_30[-min_length:],
                
                # MACD indicators
                'macd': macd[-min_length:],
                'macd_signal': macd_signal[-min_length:],
                'macd_histogram': macd_histogram[-min_length:],
                
                # Bollinger Bands
                'bb_upper': bb_upper[-min_length:],
                'bb_lower': bb_lower[-min_length:],
                'bb_width': bb_width[-min_length:],
                'bb_position': bb_position[-min_length:],
                
                # Moving averages
                'sma_10': sma_10[-min_length:],
                'sma_30': sma_30[-min_length:],
                'sma_50': sma_50[-min_length:],
                'price_to_sma10': price_to_sma10[-min_length:],
                'price_to_sma30': price_to_sma30[-min_length:],
                'price_to_sma50': price_to_sma50[-min_length:],
                
                # Other indicators
                'atr_14': atr_14[-min_length:],
                'stoch_k': stoch_k[-min_length:],
                'stoch_d': stoch_d[-min_length:],
                'volume_ratio': volume_ratio[-min_length:] if len(volume_ratio) >= min_length else np.full(min_length, np.nan)
            })
            
            technical_features.append(ticker_features)
            print(f"✅ Calculated technical indicators for {ticker}")
            
        except Exception as e:
            print(f"❌ Error calculating technical indicators for {ticker}: {e}")
    
    # Combine all ticker features
    if technical_features:
        combined_features = pd.concat(technical_features, ignore_index=True)
        
        # Save to CSV
        output_file = DATA_PROCESSED_DIR / "features_technical.csv"
        combined_features.to_csv(output_file, index=False)
        
        print(f"✅ Technical features saved to: {output_file}")
        print(f"📊 Features shape: {combined_features.shape}")
        
        return combined_features
    else:
        print("❌ No technical features calculated")
        return None

def normalize_fundamentals(fundamental_data):
    """
    Normalize and process fundamental data.
    
    This function:
    1. Handles missing values
    2. Applies log transformation to skewed variables
    3. Standardizes all features (z-score normalization)
    4. Creates sector and industry dummy variables
    
    Args:
        fundamental_data (pd.DataFrame): Raw fundamental data
    
    Returns:
        pd.DataFrame: Normalized fundamental features
    """
    print("📊 Normalizing fundamental data...")
    
    if fundamental_data is None or fundamental_data.empty:
        print("❌ No fundamental data to normalize")
        return None
    
    # Create a copy to avoid modifying original data
    df = fundamental_data.copy()
    
    # Define numerical columns for normalization
    numerical_cols = [
        'market_cap', 'pe_ratio', 'forward_pe', 'peg_ratio', 'price_to_book',
        'roe', 'roa', 'debt_to_equity', 'current_ratio', 'revenue_growth',
        'earnings_growth', 'beta'
    ]
    
    # Handle missing values by filling with median
    for col in numerical_cols:
        if col in df.columns:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
            print(f"  - Filled {df[col].isnull().sum()} missing values in {col} with median: {median_value:.3f}")
    
    # Apply log transformation to highly skewed variables
    log_transform_cols = ['market_cap', 'pe_ratio', 'forward_pe']
    for col in log_transform_cols:
        if col in df.columns:
            # Add small constant to handle zero values, then apply log
            df[f'{col}_log'] = np.log(df[col] + 1)
    
    # Create sector dummy variables
    if 'sector' in df.columns:
        sector_dummies = pd.get_dummies(df['sector'], prefix='sector')
        df = pd.concat([df, sector_dummies], axis=1)
        print(f"  - Created {len(sector_dummies.columns)} sector dummy variables")
    
    # Create industry dummy variables (top 10 most common industries)
    if 'industry' in df.columns:
        top_industries = df['industry'].value_counts().head(10).index
        for industry in top_industries:
            df[f'industry_{industry.replace(" ", "_").replace("/", "_")}'] = (df['industry'] == industry).astype(int)
        print(f"  - Created dummy variables for top 10 industries")
    
    # Standardize numerical features (z-score normalization)
    scaler = StandardScaler()
    features_to_scale = []
    
    # Include original numerical features
    for col in numerical_cols:
        if col in df.columns:
            features_to_scale.append(col)
    
    # Include log-transformed features
    for col in log_transform_cols:
        log_col = f'{col}_log'
        if log_col in df.columns:
            features_to_scale.append(log_col)
    
    if features_to_scale:
        df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
        print(f"  - Standardized {len(features_to_scale)} numerical features")
    
    # Save processed fundamental data
    output_file = DATA_PROCESSED_DIR / "features_fundamental.csv"
    df.to_csv(output_file, index=False)
    
    print(f"✅ Fundamental features saved to: {output_file}")
    print(f"📊 Features shape: {df.shape}")
    
    return df

def aggregate_sentiment(sentiment_data):
    """
    Process and aggregate sentiment and macro-economic indicators.
    
    This function:
    1. Cleans sentiment data
    2. Creates rolling averages
    3. Calculates sentiment momentum
    4. Normalizes features
    
    Args:
        sentiment_data (pd.DataFrame): Raw sentiment and macro data
    
    Returns:
        pd.DataFrame: Processed sentiment features
    """
    print("📰 Processing sentiment and macro data...")
    
    if sentiment_data is None or sentiment_data.empty:
        print("❌ No sentiment data to process")
        return None
    
    # Create a copy
    df = sentiment_data.copy()
    
    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Calculate rolling averages for VIX (fear index)
    df['vix_ma_7'] = df['vix'].rolling(window=7).mean()
    df['vix_ma_30'] = df['vix'].rolling(window=30).mean()
    df['vix_momentum'] = df['vix'] / df['vix_ma_30'] - 1  # VIX relative to 30-day average
    
    # Create VIX categories (low, medium, high fear)
    vix_quantiles = df['vix'].quantile([0.33, 0.67])
    df['vix_category'] = pd.cut(df['vix'], 
                               bins=[-np.inf, vix_quantiles[0.33], vix_quantiles[0.67], np.inf],
                               labels=['low_fear', 'medium_fear', 'high_fear'])
    
    # Process sentiment scores
    if 'market_sentiment' in df.columns:
        df['sentiment_ma_7'] = df['market_sentiment'].rolling(window=7).mean()
        df['sentiment_ma_30'] = df['market_sentiment'].rolling(window=30).mean()
        df['sentiment_momentum'] = df['market_sentiment'] - df['sentiment_ma_30']
    
    if 'news_sentiment' in df.columns:
        df['news_sentiment_ma_7'] = df['news_sentiment'].rolling(window=7).mean()
        df['news_sentiment_ma_30'] = df['news_sentiment'].rolling(window=30).mean()
    
    # Create dummy variables for VIX categories
    vix_dummies = pd.get_dummies(df['vix_category'], prefix='vix')
    df = pd.concat([df, vix_dummies], axis=1)
    
    # Normalize numerical features
    numerical_cols = ['vix', 'vix_ma_7', 'vix_ma_30', 'vix_momentum']
    if 'market_sentiment' in df.columns:
        numerical_cols.extend(['market_sentiment', 'sentiment_ma_7', 'sentiment_ma_30', 'sentiment_momentum'])
    if 'news_sentiment' in df.columns:
        numerical_cols.extend(['news_sentiment', 'news_sentiment_ma_7', 'news_sentiment_ma_30'])
    
    # Standardize features
    scaler = StandardScaler()
    existing_cols = [col for col in numerical_cols if col in df.columns]
    if existing_cols:
        df[existing_cols] = scaler.fit_transform(df[existing_cols])
        print(f"  - Standardized {len(existing_cols)} sentiment features")
    
    # Save processed sentiment data
    output_file = DATA_PROCESSED_DIR / "features_sentiment.csv"
    df.to_csv(output_file, index=False)
    
    print(f"✅ Sentiment features saved to: {output_file}")
    print(f"📊 Features shape: {df.shape}")
    
    return df

def main():
    """
    Main function to execute the feature engineering pipeline.
    """
    print("🚀 Starting Phase 1: Feature Engineering")
    print("=" * 50)
    
    # Ensure directories exist
    ensure_processed_directories()
    
    try:
        # Load raw data
        print("📁 Loading raw data...")
        
        # Load stock price data
        stock_file = DATA_RAW_DIR / "stock_prices_ohlcv_raw.csv"
        if stock_file.exists():
            stock_data = pd.read_csv(stock_file, index_col=0, parse_dates=True, header=[0, 1])
            print(f"✅ Loaded stock data: {stock_data.shape}")
        else:
            print(f"❌ Stock data file not found: {stock_file}")
            stock_data = None
        
        # Load fundamental data
        fundamental_file = DATA_RAW_DIR / "fundamentals_raw.csv"
        if fundamental_file.exists():
            fundamental_data = pd.read_csv(fundamental_file)
            print(f"✅ Loaded fundamental data: {fundamental_data.shape}")
        else:
            print(f"❌ Fundamental data file not found: {fundamental_file}")
            fundamental_data = None
        
        # Load sentiment data
        sentiment_file = DATA_RAW_DIR / "sentiment_macro_raw.csv"
        if sentiment_file.exists():
            sentiment_data = pd.read_csv(sentiment_file)
            print(f"✅ Loaded sentiment data: {sentiment_data.shape}")
        else:
            print(f"❌ Sentiment data file not found: {sentiment_file}")
            sentiment_data = None
        
        print("\n" + "-" * 30)
        
        # Process each type of data
        if stock_data is not None:
            technical_features = calculate_technical_indicators(stock_data)
        
        if fundamental_data is not None:
            fundamental_features = normalize_fundamentals(fundamental_data)
        
        if sentiment_data is not None:
            sentiment_features = aggregate_sentiment(sentiment_data)
        
        print("\n" + "=" * 50)
        print("✅ Phase 1: Feature Engineering Complete!")
        print(f"📁 All processed features saved to: {DATA_PROCESSED_DIR}")
        
    except Exception as e:
        print(f"❌ Error in feature engineering pipeline: {e}")

if __name__ == "__main__":
    main()
