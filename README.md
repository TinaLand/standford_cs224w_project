# CS224W Stock RL GNN Project

## 📋 Project Overview

This project implements a Graph Neural Network (GNN) approach for stock market prediction and reinforcement learning-based trading. The project is structured in phases, starting with data collection and feature engineering.

**Current Status:** Phase 1 - Data Collection & Feature Engineering ✅

## 🗂️ Project Structure

```
CS224W_Stock_RL_GNN/
├── data/
│   ├── raw/                          # Raw data files
│   │   ├── stock_prices_ohlcv_raw.csv    # OHLCV stock price data (2015-2025)
│   │   ├── fundamentals_raw.csv          # Fundamental metrics (P/E, ROE, etc.)
│   │   └── sentiment_macro_raw.csv       # Sentiment and macro indicators
│   ├── processed/                    # Processed feature files
│   │   ├── features_technical.csv        # Technical indicators (RSI, MACD, etc.)
│   │   ├── features_fundamental.csv      # Normalized fundamental features
│   │   └── features_sentiment.csv        # Processed sentiment features
│   └── edges/                        # Edge parameter files for graph construction
│       ├── edges_dynamic_corr_params.csv   # Rolling correlation parameters
│       ├── edges_dynamic_fund_sim_params.csv # Fundamental similarity parameters
│       └── edges_sector_connections.csv     # Sector-based connections
├── scripts/
│   ├── phase1_data_collection.py          # Data download and collection
│   ├── phase1_feature_engineering.py      # Feature calculation (with TA-Lib)
│   ├── phase1_feature_engineering_simple.py # Feature calculation (no TA-Lib)
│   ├── phase1_edge_parameter_calc.py      # Edge parameter computation
│   └── utils_data.py                      # Data utility functions
├── requirements.txt                       # Python dependencies (with TA-Lib)
├── requirements_simple.txt               # Python dependencies (no TA-Lib)
└── README.md                             # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+ 
- pip or conda package manager

### Installation

1. **Clone the repository** (if using git):
   ```bash
   git clone <repository-url>
   cd cs224_porject
   ```

2. **Create and activate virtual environment** (recommended):
   ```bash
   # create virtual env
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:

   **Option A: Full Version** (includes TA-Lib, full functionality):
   ```bash
   pip install -r requirements.txt
   ```
   
   **Option B: Simple Version** (no TA-Lib dependency, recommended for quick start):
   ```bash
   pip install -r requirements_simple.txt
   ```

   **TA-Lib Installation Notes** (only needed for Option A):
   - **MacOS**: `brew install ta-lib` then `pip install TA-Lib`
   - **Ubuntu**: `sudo apt-get install libta-lib-dev` then `pip install TA-Lib`
   - **Windows**: Download pre-compiled wheels from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)

### Running Phase 1

Execute the Phase 1 scripts in order:

1. **Data Collection** [[memory:3128469]]:
   ```bash
   cd scripts
   python phase1_data_collection.py
   ```
   - Downloads OHLCV data for selected stocks
   - Collects fundamental data (simulated)
   - Gathers sentiment and VIX data

2. **Feature Engineering**:
   
   **Option A: Full Version** (if TA-Lib is installed):
   ```bash
   python phase1_feature_engineering.py
   ```
   
   **Option B: Simple Version** (recommended, no TA-Lib needed):
   ```bash
   python phase1_feature_engineering_simple.py
   ```
   
   Features include:
   - Calculates technical indicators (RSI, MACD, Bollinger Bands)
   - Normalizes fundamental features
   - Processes sentiment data

3. **Edge Parameter Calculation**:
   ```bash
   python phase1_edge_parameter_calc.py
   ```
   - Computes rolling correlations between stocks
   - Calculates fundamental similarity measures
   - Creates sector-based connection parameters

## 📊 Data Description

### Stock Selection
- **Default**: Top 50 holdings from SPY ETF
- **Configurable**: Modify `CONFIG` in `phase1_data_collection.py`
- **Date Range**: 2015-2025 (configurable)

### Features Generated

#### Technical Features
- **Price-based**: Returns, log returns, volatility
- **Momentum**: RSI (14-day, 30-day), Stochastic oscillator
- **Trend**: MACD, Moving averages (SMA, EMA)
- **Volatility**: Bollinger Bands, ATR
- **Volume**: Volume ratios and moving averages

#### Fundamental Features
- **Valuation**: P/E ratio, Price-to-Book, PEG ratio
- **Profitability**: ROE, ROA, profit margins
- **Financial Health**: Debt-to-Equity, Current ratio
- **Growth**: Revenue growth, earnings growth
- **Risk**: Beta coefficient

#### Sentiment & Macro Features
- **Market Sentiment**: VIX (fear index), rolling averages
- **News Sentiment**: Aggregated sentiment scores (simulated)
- **Macro Indicators**: Various economic indicators

### Edge Parameters
- **Dynamic Correlations**: 30-day rolling correlations between stock returns
- **Fundamental Similarity**: Cosine similarity of fundamental metrics
- **Sector Connections**: Industry and sector-based relationships

## 🛠️ Configuration

Modify settings in `scripts/phase1_data_collection.py`:

```python
CONFIG = {
    'START_DATE': '2015-01-01',    # Data start date
    'END_DATE': '2025-01-01',      # Data end date
    'STOCK_SOURCE': 'SPY',         # 'SPY', 'QQQ', or 'CUSTOM'
    'NUM_STOCKS': 50,              # Number of stocks to include
    'CUSTOM_TICKERS': ['AAPL', 'MSFT', ...]  # Custom ticker list
}
```

## 📈 Usage Examples

### Loading Processed Data
```python
from scripts.utils_data import load_data_file

# Load technical features
technical_data = load_data_file('features_technical.csv')

# Load correlation parameters
correlations = load_data_file('edges_dynamic_corr_params.csv', directory='edges')
```

### Data Validation
```python
from scripts.utils_data import validate_ticker_data

# Validate data quality
results = validate_ticker_data(
    technical_data, 
    required_columns=['ticker', 'date', 'returns'],
    min_observations=100
)
```

## 🔮 Upcoming Phases

- **Phase 2**: Graph Construction - Build dynamic stock graphs
- **Phase 3**: Baseline GNN Training - Train baseline GNN models  
- **Phase 4**: GTransformer PEARL Training - Advanced architectures
- **Phase 5**: RL Integration - Reinforcement learning for trading
- **Phase 6**: Evaluation & Visualization - Performance analysis
- **Phase 7**: Optimization & Extension - Model improvements

## 📋 Mathematical Foundation

### Technical Indicators
- **RSI**: $RSI = 100 - \frac{100}{1 + RS}$ where $RS = \frac{\text{Average Gain}}{\text{Average Loss}}$
- **MACD**: $MACD = EMA_{12} - EMA_{26}$
- **Bollinger Bands**: $BB_{upper/lower} = SMA_n \pm k \cdot \sigma_n$

### Edge Weights
- **Correlation**: $\rho_{ij}(t) = \frac{Cov(r_{i,t}, r_{j,t})}{\sigma_{i,t} \sigma_{j,t}}$
- **Cosine Similarity**: $cos(\theta) = \frac{\mathbf{f_i} \cdot \mathbf{f_j}}{|\mathbf{f_i}||\mathbf{f_j}|}$

## 🤝 Contributing

1. Follow the established code structure and documentation standards [[memory:3128464]]
2. Add comprehensive comments explaining mathematical concepts
3. Include unit tests for new functions
4. Update this README when adding new phases

## 📝 Notes

- All functions requiring implementation are marked with 'start code' and 'end code' comments [[memory:3128462]]
- Comments and documentation are in English [[memory:3128459]] [[memory:2522995]]
- Mathematical explanations are provided for educational purposes [[memory:3128464]]

## 🐛 Troubleshooting

### Common Issues

1. **TA-Lib installation problems**: 
   - **Quick solution**: Use the simple version: `python phase1_feature_engineering_simple.py`
   - **Full solution**: Follow platform-specific TA-Lib installation steps above
   - The simple version provides equivalent functionality without TA-Lib dependency

2. **Virtual environment activation**:
   - Ensure virtual environment is activated before installing packages
   - Windows users: Try different activation commands (PowerShell vs Command Prompt)
   - If activation fails, create a new virtual environment

3. **VIX data download**: 
   - Ensure internet connection and try running again
   - Script includes fallback to dummy data if VIX download fails

4. **Memory issues**: 
   - Reduce the number of stocks in CONFIG if running on limited memory
   - Use smaller date ranges for initial testing

5. **Module import errors**:
   - Verify virtual environment is activated
   - Reinstall requirements: `pip install -r requirements_simple.txt`

### Getting Help
- Check the validation output from `utils_data.py` functions
- Ensure all data directories exist before running scripts
- Verify date ranges are valid for market data availability  
- Try the simple version first before troubleshooting TA-Lib issues