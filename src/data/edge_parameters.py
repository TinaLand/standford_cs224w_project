"""
Script for calculating dynamic edge parameters (e.g., rolling correlation, cosine similarity)

Phase 1: Edge Parameter Calculation Script
=========================================
This script calculates dynamic edge parameters for the stock graph based on:
1. Rolling correlations between stock returns
2. Fundamental similarity measures
3. Sector/industry relationships

These parameters will be used in Phase 2 to construct dynamic graph edges.

Key Functions:
- compute_rolling_correlation(): Calculates time-varying correlations
- compute_fundamental_similarity(): Computes cosine similarity for fundamentals
- compute_sector_similarity(): Creates sector-based connections
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import itertools
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Set up paths
# NOTE: This file lives in `src/data/`, so the project root is three levels up.
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
DATA_EDGES_DIR = BASE_DIR / "data" / "edges"

def ensure_edges_directories():
    """
    Create edges data directories if they don't exist.
    """
    DATA_EDGES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✅ Edges data directory ready: {DATA_EDGES_DIR}")

def compute_rolling_correlation(technical_data, window=30, min_periods=20):
    """
    Compute rolling correlation between stock returns over time.
    
    This creates time-varying edge weights based on how similarly stocks move.
    Higher correlation indicates stronger connection between stocks.
    
    Mathematical Background:
    - Pearson correlation coefficient r = Cov(X,Y) / (σ_X * σ_Y)
    - Rolling window captures time-varying relationships
    - Values range from -1 (perfect negative correlation) to +1 (perfect positive correlation)
    
    Args:
        technical_data (pd.DataFrame): Technical features with returns data
        window (int): Rolling window size in days (default: 30)
        min_periods (int): Minimum observations required (default: 20)
    
    Returns:
        pd.DataFrame: Time-varying correlation matrix with columns [date, ticker1, ticker2, correlation]
    """
    print(f"📈 Computing rolling correlations with window={window} days...")
    
    if technical_data is None or technical_data.empty:
        print("❌ No technical data available for correlation calculation")
        return None
    
    # Pivot data to have tickers as columns and dates as rows
    returns_pivot = technical_data.pivot(index='Date', columns='ticker', values='returns')
    
    # Remove any columns (tickers) with insufficient data
    valid_tickers = returns_pivot.columns[returns_pivot.count() >= min_periods * 2]
    returns_pivot = returns_pivot[valid_tickers]
    
    print(f"  - Using {len(valid_tickers)} tickers with sufficient data")
    print(f"  - Date range: {returns_pivot.index.min()} to {returns_pivot.index.max()}")
    
    correlation_data = []
    
    # Calculate rolling correlations for each pair of stocks
    ticker_pairs = list(itertools.combinations(valid_tickers, 2))
    print(f"  - Computing correlations for {len(ticker_pairs)} stock pairs...")
    
    for i, (ticker1, ticker2) in enumerate(ticker_pairs):
        if i % 50 == 0:
            print(f"    Progress: {i}/{len(ticker_pairs)} pairs completed")
        
        # Get returns series for both tickers
        returns1 = returns_pivot[ticker1].dropna()
        returns2 = returns_pivot[ticker2].dropna()
        
        # Find common dates
        common_dates = returns1.index.intersection(returns2.index)
        if len(common_dates) < min_periods:
            continue
        
        # Align the series
        aligned_returns1 = returns1.loc[common_dates]
        aligned_returns2 = returns2.loc[common_dates]
        
        # Calculate rolling correlation
        rolling_corr = aligned_returns1.rolling(window=window, min_periods=min_periods).corr(aligned_returns2)
        
        # Store results
        for date, correlation in rolling_corr.dropna().items():
            correlation_data.append({
                'Date': date,
                'ticker1': ticker1,
                'ticker2': ticker2,
                'correlation': correlation,
                'abs_correlation': abs(correlation),  # Absolute correlation for edge strength
                'correlation_strength': 'strong' if abs(correlation) > 0.7 else 'moderate' if abs(correlation) > 0.3 else 'weak'
            })
    
    # Convert to DataFrame
    correlation_df = pd.DataFrame(correlation_data)
    
    if not correlation_df.empty:
        # Add some summary statistics
        print(f"  - Correlation statistics:")
        print(f"    Mean absolute correlation: {correlation_df['abs_correlation'].mean():.3f}")
        print(f"    Strong correlations (|r| > 0.7): {(correlation_df['abs_correlation'] > 0.7).sum()} pairs")
        print(f"    Moderate correlations (0.3 < |r| < 0.7): {((correlation_df['abs_correlation'] > 0.3) & (correlation_df['abs_correlation'] <= 0.7)).sum()} pairs")
        
        # Save to CSV
        output_file = DATA_EDGES_DIR / "edges_dynamic_corr_params.csv"
        correlation_df.to_csv(output_file, index=False)
        
        print(f"✅ Correlation data saved to: {output_file}")
        print(f"📊 Data shape: {correlation_df.shape}")
        
        return correlation_df
    else:
        print("❌ No correlation data generated")
        return None

def compute_fundamental_similarity(fundamental_data):
    """
    Compute similarity between stocks based on fundamental metrics.
    
    This creates static edge weights based on how similar companies are
    in terms of their financial characteristics.
    
    Mathematical Background:
    - Cosine similarity: cos(θ) = (A·B) / (||A|| ||B||)
    - Measures angle between feature vectors
    - Values range from 0 (orthogonal) to 1 (identical direction)
    
    Args:
        fundamental_data (pd.DataFrame): Normalized fundamental features
    
    Returns:
        pd.DataFrame: Pairwise fundamental similarity with columns [ticker1, ticker2, similarity]
    """
    print("📊 Computing fundamental similarity...")
    
    if fundamental_data is None or fundamental_data.empty:
        print("❌ No fundamental data available for similarity calculation")
        return None
    
    # Select numerical features for similarity calculation
    feature_cols = []
    potential_features = [
        'market_cap', 'pe_ratio', 'forward_pe', 'peg_ratio', 'price_to_book',
        'roe', 'roa', 'debt_to_equity', 'current_ratio', 'revenue_growth',
        'earnings_growth', 'beta', 'market_cap_log', 'pe_ratio_log', 'forward_pe_log'
    ]
    
    for col in potential_features:
        if col in fundamental_data.columns:
            feature_cols.append(col)
    
    print(f"  - Using {len(feature_cols)} fundamental features")
    
    # Create feature matrix (tickers x features)
    feature_matrix = fundamental_data.set_index('ticker')[feature_cols].fillna(0)
    
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(feature_matrix.values)
    
    # Convert to DataFrame with ticker labels
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=feature_matrix.index,
        columns=feature_matrix.index
    )
    
    # Convert to pairwise format
    similarity_data = []
    tickers = feature_matrix.index.tolist()
    
    for i, ticker1 in enumerate(tickers):
        for j, ticker2 in enumerate(tickers):
            if i < j:  # Only store upper triangle (avoid duplicates)
                similarity = similarity_df.loc[ticker1, ticker2]
                similarity_data.append({
                    'ticker1': ticker1,
                    'ticker2': ticker2,
                    'fundamental_similarity': similarity,
                    'similarity_strength': 'high' if similarity > 0.8 else 'medium' if similarity > 0.5 else 'low'
                })
    
    similarity_pairs_df = pd.DataFrame(similarity_data)
    
    if not similarity_pairs_df.empty:
        # Add summary statistics
        print(f"  - Similarity statistics:")
        print(f"    Mean similarity: {similarity_pairs_df['fundamental_similarity'].mean():.3f}")
        print(f"    High similarity (>0.8): {(similarity_pairs_df['fundamental_similarity'] > 0.8).sum()} pairs")
        print(f"    Medium similarity (0.5-0.8): {((similarity_pairs_df['fundamental_similarity'] > 0.5) & (similarity_pairs_df['fundamental_similarity'] <= 0.8)).sum()} pairs")
        
        # Also add sector information for enhanced similarity
        if 'sector' in fundamental_data.columns:
            sector_dict = fundamental_data.set_index('ticker')['sector'].to_dict()
            similarity_pairs_df['same_sector'] = similarity_pairs_df.apply(
                lambda row: sector_dict.get(row['ticker1']) == sector_dict.get(row['ticker2']), axis=1
            )
            print(f"    Same sector pairs: {similarity_pairs_df['same_sector'].sum()} pairs")
        
        # Save to CSV
        output_file = DATA_EDGES_DIR / "edges_dynamic_fund_sim_params.csv"
        similarity_pairs_df.to_csv(output_file, index=False)
        
        print(f"✅ Fundamental similarity data saved to: {output_file}")
        print(f"📊 Data shape: {similarity_pairs_df.shape}")
        
        return similarity_pairs_df
    else:
        print("❌ No fundamental similarity data generated")
        return None

def compute_sector_similarity(fundamental_data):
    """
    Compute sector-based connections between stocks.
    
    This creates categorical edge connections based on industry sectors.
    Stocks in the same sector get higher connection weights.
    
    Args:
        fundamental_data (pd.DataFrame): Fundamental data with sector information
    
    Returns:
        pd.DataFrame: Sector-based connections with columns [ticker1, ticker2, sector_connection, weight]
    """
    print("🏭 Computing sector-based connections...")
    
    # Load sector data from static file
    # BASE_DIR is already the project root (parent.parent.parent from src/data/edge_parameters.py)
    sector_file = BASE_DIR / "data" / "raw" / "static_sector_industry.csv"
    sector_dict_upper = {}
    sector_dict_original = {}
    
    if sector_file.exists():
        try:
            sector_df = pd.read_csv(sector_file)
            if 'Ticker' in sector_df.columns and 'Sector' in sector_df.columns:
                # Create a case-insensitive mapping (convert both to uppercase for matching)
                sector_df['Ticker_Upper'] = sector_df['Ticker'].str.upper()
                sector_dict_upper = sector_df.set_index('Ticker_Upper')['Sector'].to_dict()
                # Also create original case mapping for reference
                sector_dict_original = sector_df.set_index('Ticker')['Sector'].to_dict()
                
                print(f"✅ Loaded sector data from static file: {len(sector_dict_upper)} tickers")
        except Exception as e:
            print(f"⚠️  Could not load sector data: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠️  Sector file not found: {sector_file}")
    
    # Check if fundamental_data is available
    if fundamental_data is None or fundamental_data.empty:
        print("❌ No fundamental data available")
        return None
    
    # Try to add sector column if not present
    if 'sector' not in fundamental_data.columns:
        if sector_dict_upper:
            # Map sector data using case-insensitive matching
            fundamental_data['ticker_upper'] = fundamental_data['ticker'].str.upper()
            fundamental_data['sector'] = fundamental_data['ticker_upper'].map(sector_dict_upper)
            fundamental_data = fundamental_data.drop('ticker_upper', axis=1)
            matched_count = fundamental_data['sector'].notna().sum()
            total_count = len(fundamental_data)
            print(f"✅ Mapped sector data: {matched_count}/{total_count} tickers matched")
            if matched_count == 0:
                print(f"⚠️  Warning: No tickers matched. Sample tickers in fundamental_data: {fundamental_data['ticker'].head(5).tolist()}")
                if sector_dict_original:
                    print(f"⚠️  Sample tickers in sector file: {list(sector_dict_original.keys())[:5]}")
        else:
            print("❌ No sector data available to map")
            return None
    
    # Check if sector data is available
    if 'sector' not in fundamental_data.columns:
        print("❌ No sector column in fundamental data")
        return None
    
    if fundamental_data['sector'].isna().all():
        print("❌ All sector values are NaN (ticker names may not match)")
        print(f"   Sample tickers: {fundamental_data['ticker'].head(10).tolist()}")
        return None
    
    sector_data = []
    tickers = fundamental_data['ticker'].tolist()
    sector_dict = fundamental_data.set_index('ticker')['sector'].to_dict()
    
    # Create sector connections
    for i, ticker1 in enumerate(tickers):
        for j, ticker2 in enumerate(tickers):
            if i < j:  # Avoid duplicates
                sector1 = sector_dict.get(ticker1, 'Unknown')
                sector2 = sector_dict.get(ticker2, 'Unknown')
                
                if sector1 == sector2 and sector1 != 'Unknown':
                    # Same sector - strong connection
                    connection_type = 'same_sector'
                    weight = 1.0
                elif sector1 != 'Unknown' and sector2 != 'Unknown':
                    # Different sectors - weak connection
                    connection_type = 'different_sector'
                    weight = 0.1
                else:
                    # Unknown sector - no connection
                    connection_type = 'unknown'
                    weight = 0.0
                
                sector_data.append({
                    'ticker1': ticker1,
                    'ticker2': ticker2,
                    'sector1': sector1,
                    'sector2': sector2,
                    'sector_connection': connection_type,
                    'sector_weight': weight
                })
    
    sector_df = pd.DataFrame(sector_data)
    
    if not sector_df.empty:
        # Summary statistics
        print(f"  - Sector connection statistics:")
        same_sector_count = (sector_df['sector_connection'] == 'same_sector').sum()
        different_sector_count = (sector_df['sector_connection'] == 'different_sector').sum()
        print(f"    Same sector pairs: {same_sector_count}")
        print(f"    Different sector pairs: {different_sector_count}")
        
        # Count stocks per sector
        sector_counts = fundamental_data['sector'].value_counts()
        print(f"    Sectors represented: {len(sector_counts)}")
        print(f"    Top 3 sectors: {sector_counts.head(3).to_dict()}")
        
        # Save to CSV
        output_file = DATA_EDGES_DIR / "edges_sector_connections.csv"
        sector_df.to_csv(output_file, index=False)
        
        print(f"✅ Sector connection data saved to: {output_file}")
        print(f"📊 Data shape: {sector_df.shape}")
        
        return sector_df
    else:
        print("❌ No sector connection data generated")
        return None

def main():
    """
    Main function to execute the edge parameter calculation pipeline.
    """
    print("🚀 Starting Phase 1: Edge Parameter Calculation")
    print("=" * 50)
    
    # Ensure directories exist
    ensure_edges_directories()
    
    try:
        # Load processed data
        print("📁 Loading processed data...")
        
        # Load consolidated node features (contains technical + fundamental + sentiment data)
        consolidated_file = DATA_PROCESSED_DIR / "node_features_X_t_final.csv"
        if consolidated_file.exists():
            consolidated_data = pd.read_csv(consolidated_file, index_col='Date', parse_dates=True)
            print(f"✅ Loaded consolidated data: {consolidated_data.shape}")
            
            # Extract technical indicators (log returns) from consolidated data
            technical_cols = [col for col in consolidated_data.columns if 'LogRet_1d_' in col]
            if technical_cols:
                # Create long format data for correlation calculation
                technical_data_list = []
                for col in technical_cols:
                    ticker = col.split('_')[-1]  # Extract ticker from column name
                    temp_df = consolidated_data[[col]].reset_index()
                    temp_df.columns = ['Date', 'returns']
                    temp_df['ticker'] = ticker
                    technical_data_list.append(temp_df)
                
                technical_data = pd.concat(technical_data_list, ignore_index=True)
                technical_data['Date'] = pd.to_datetime(technical_data['Date'])
                print(f"✅ Extracted technical features for correlation: {technical_data.shape}")
            else:
                print("❌ No technical indicators found in consolidated data")
                technical_data = None
        else:
            print(f"❌ Consolidated data file not found: {consolidated_file}")
            technical_data = None
        
        # Extract fundamental features from consolidated data
        if 'consolidated_data' in locals() and consolidated_data is not None:
            # Extract fundamental columns (PE, ROE ratios) - exclude technical indicators  
            pe_cols = [col for col in consolidated_data.columns if col.endswith('_PE') and not '_PE_Log' in col]
            roe_cols = [col for col in consolidated_data.columns if col.endswith('_ROE') and not '_ROE_Log' in col]
            
            if pe_cols or roe_cols:
                # Transform wide format to long format for fundamental similarity calculation
                fundamental_data_list = []
                
                # Get the latest date's fundamental data (fundamentals don't change daily)
                latest_fundamentals = consolidated_data.iloc[-1]
                
                # Extract tickers and their fundamental metrics
                tickers_with_fundamentals = set()
                if pe_cols:
                    tickers_with_fundamentals.update([col.split('_PE')[0] for col in pe_cols])
                if roe_cols:
                    tickers_with_fundamentals.update([col.split('_ROE')[0] for col in roe_cols])
                
                for ticker in tickers_with_fundamentals:
                    row_data = {'ticker': ticker}
                    
                    # Add PE ratio if available
                    pe_col = f'{ticker}_PE'
                    if pe_col in latest_fundamentals.index:
                        row_data['pe_ratio'] = latest_fundamentals[pe_col]
                    
                    # Add ROE if available  
                    roe_col = f'{ticker}_ROE'
                    if roe_col in latest_fundamentals.index:
                        row_data['roe'] = latest_fundamentals[roe_col]
                    
                    fundamental_data_list.append(row_data)
                
                if fundamental_data_list:
                    fundamental_data = pd.DataFrame(fundamental_data_list)
                    print(f"✅ Transformed fundamental features for {len(fundamental_data_list)} tickers: {fundamental_data.shape}")
                else:
                    print("❌ No valid fundamental features found")
                    fundamental_data = None
            else:
                print("❌ No fundamental features found in consolidated data")
                fundamental_data = None
                
            # Extract sentiment/macro features from consolidated data
            sentiment_cols = [col for col in consolidated_data.columns if any(metric in col for metric in ['VIX', 'Sentiment'])]
            if sentiment_cols:
                sentiment_data = consolidated_data[sentiment_cols].reset_index()
                print(f"✅ Extracted sentiment/macro features: {sentiment_data.shape}")
            else:
                print("❌ No sentiment/macro features found in consolidated data")
                sentiment_data = None
        else:
            fundamental_data = None
            sentiment_data = None
        
        print("\n" + "-" * 30)
        
        # Compute different types of edge parameters
        if technical_data is not None:
            print("1️⃣ Computing rolling correlations...")
            correlation_data = compute_rolling_correlation(technical_data, window=30)
        
        if fundamental_data is not None:
            print("\n2️⃣ Computing fundamental similarity...")
            similarity_data = compute_fundamental_similarity(fundamental_data)
            
            print("\n3️⃣ Computing sector connections...")
            sector_data = compute_sector_similarity(fundamental_data)
        
        print("\n" + "=" * 50)
        print("✅ Phase 1: Edge Parameter Calculation Complete!")
        print(f"📁 All edge parameters saved to: {DATA_EDGES_DIR}")
        print("\n📋 Generated edge parameter files:")
        
        # List generated files
        edge_files = list(DATA_EDGES_DIR.glob("*.csv"))
        for edge_file in edge_files:
            print(f"  - {edge_file.name}")
        
    except Exception as e:
        print(f"❌ Error in edge parameter calculation pipeline: {e}")

if __name__ == "__main__":
    main()
