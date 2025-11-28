# phase1_static_data_collection.py
import pandas as pd
import numpy as np
import os
from pathlib import Path
import random

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
# Ensure the raw data directory exists
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)


# --- Helper Function ---

def get_ticker_list():
    """
    Reads the list of tickers from the OHLCV raw data file generated 
    in phase1_data_collection.py.
    """
    ohlcv_file = DATA_RAW_DIR / 'stock_prices_ohlcv_raw.csv'
    if not ohlcv_file.exists():
        print(f"❌ Error: OHLCV raw data not found at {ohlcv_file}. Please run phase1_data_collection.py first.")
        return []
    
    # Load just the header to extract tickers
    df = pd.read_csv(ohlcv_file, index_col=0, nrows=0)
    
    # Assuming columns follow the pattern 'Feature_TICKER'
    tickers = sorted(list(set(col.split('_')[-1] for col in df.columns if '_' in col)))
    
    if len(tickers) < 2:
        fallback_tickers = [
            'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'BRK-B', 'LLY', 'TSLA', 'V',
            'JPM', 'XOM', 'JNJ', 'WMT', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV',
            'KO', 'PEP', 'AVGO', 'COST', 'PFE', 'ADBE', 'CSCO', 'CMCSA', 'NFLX', 'DIS',
            'ACN', 'CRM', 'TMO', 'QCOM', 'TXN', 'UNH', 'BAC', 'MCD', 'ORCL', 'INTC',
            'SBUX', 'CAT', 'GE', 'NKE', 'AXP', 'IBM', 'MMM', 'VZ', 'FDX', 'GOOG'
        ]
        print("⚠️  Warning: Unable to infer ticker list from OHLCV data. Falling back to default SPY top-50 list.")
        tickers = fallback_tickers
    
    return tickers


# --- Static Data Collection Functions ---

def download_sector_industry_data(tickers, output_path):
    """
    Simulates the collection of Sector/Industry classification data[cite: 36, 57].
    
    Saves to: static_sector_industry.csv
    """
    print(f"\n--- 1. Simulating Sector/Industry Data for {len(tickers)} stocks... ---")
    
    # Define a set of standard sectors and industries for simulation
    sectors = ['Technology', 'Financials', 'Healthcare', 'Energy', 'Consumer Discretionary']
    industries_map = {
        'Technology': ['Software', 'Hardware', 'Semiconductors'],
        'Financials': ['Banking', 'Insurance'],
        'Healthcare': ['Pharmaceuticals', 'Biotech'],
        'Energy': ['Oil & Gas', 'Renewable Energy'],
        'Consumer Discretionary': ['E-commerce', 'Automobiles']
    }
    
    data = []
    # Assign a simulated sector and industry to each ticker
    for ticker in tickers:
        sector = random.choice(sectors)
        industry = random.choice(industries_map[sector])
        data.append({'Ticker': ticker, 'Sector': sector, 'Industry': industry})
        
    sector_df = pd.DataFrame(data)
    file_path = output_path / 'static_sector_industry.csv'
    sector_df.to_csv(file_path, index=False)
    print(f"✅ Simulated Sector/Industry data saved to: {file_path}")
    print(f"Rationale: Needed for Static Edges (Weight: 1.0 same industry, 0.5 same sector)[cite: 57].")


def download_supply_chain_competitor_data(tickers, output_path):
    """
    Simulates the collection of Supply Chain and Competitor relationships[cite: 37, 67, 71].
    
    Saves to: static_supply_competitor_edges.csv
    """
    print(f"\n--- 2. Simulating Supply Chain and Competitor Data... ---")
    
    edges = []
    
    # Simulate Supply Chain: Customer -> Supplier (Binary edge [cite: 68])
    # Assume 10-15 random customer-supplier links
    num_supply_links = int(len(tickers) * 0.5) if len(tickers) > 10 else 5
    for _ in range(num_supply_links):
        customer = random.choice(tickers)
        supplier = random.choice([t for t in tickers if t != customer])
        edges.append({'Ticker1': customer, 'Ticker2': supplier, 'Relation': 'SUPPLY_CHAIN', 'Weight': 1.0})
    print(f"  - Simulated {num_supply_links} Supply Chain (Customer->Supplier) links[cite: 67].")

    # Simulate Competitor: Competitor <-> Competitor (Binary edge [cite: 72])
    # Assume 5-10 random competitor links
    num_comp_links = int(len(tickers) * 0.3) if len(tickers) > 10 else 3
    for _ in range(num_comp_links):
        comp1 = random.choice(tickers)
        comp2 = random.choice([t for t in tickers if t != comp1])
        # Add both directions for undirected competitor relationship
        edges.append({'Ticker1': comp1, 'Ticker2': comp2, 'Relation': 'COMPETITOR', 'Weight': 1.0})
        edges.append({'Ticker1': comp2, 'Ticker2': comp1, 'Relation': 'COMPETITOR', 'Weight': 1.0})
    print(f"  - Simulated {num_comp_links * 2} Competitor links (undirected)[cite: 71].")

    
    edges_df = pd.DataFrame(edges)
    file_path = output_path / 'static_supply_competitor_edges.csv'
    edges_df.to_csv(file_path, index=False)
    print(f"✅ Simulated Static Edge data saved to: {file_path}")


def main():
    """
    Main execution function to run all static data collection steps.
    """
    print(f"Starting Phase 1: Static Edge Data Collection")
    print("=" * 50)
    
    tickers = get_ticker_list()
    if not tickers:
        return
        
    # Step 1: Sector and Industry Classification
    download_sector_industry_data(tickers, DATA_RAW_DIR)
    
    # Step 2: Supply Chain and Competitor Relationships
    download_supply_chain_competitor_data(tickers, DATA_RAW_DIR)

    print("\nPhase 1 Static Data Collection complete. Data is in 'data/raw/'.")

if __name__ == "__main__":
    main()