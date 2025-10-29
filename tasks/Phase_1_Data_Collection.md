# 📊 Phase 1: Data Collection & Feature Engineering

**Objective:** Prepare the multi-modal input data $G_{t}=(V,E_{t},X_{t})$ for graph construction.

## 1.1 Raw Data Acquisition

* **OHLCV (Open, High, Low, Close, Volume):** Collect daily stock price and volume data (covering 2015-2025).
* **Fundamental Data (Fundamentals):** Gather financial metrics such as Market Cap, P/E, P/B, ROE, Debt/Equity, etc..
* **News and Sentiment Data:** Acquire market and individual stock news polarity and social media mentions.
* **Macroeconomic Data (Macro):** Collect sector index returns, interest rates, VIX, and revenue exposure to major customers.

## 1.2 Node Feature ($X_{t}$) Computation

These features will be assigned to each stock node.

| Feature Category | Specific Indicators | Source |
| :--- | :--- | :--- |
| **Technical Indicators** | Returns (1-, 5-, 20-day), MA20/50, Volatility, RSI, MACD, Bollinger Band width, ATR, OBV, Trading volume, Spikes | OHLCV |
| **Fundamentals** | Market Cap, P/E, P/B, EV/EBITDA, ROE, EPS growth, Debt/Equity, Current Ratio, Beta | Fundamental Data |
| **Sentiment Features** | News polarity, Social media mentions | Sentiment Data |
| **Macro/Supply Chain** | Sector index returns, Interest rates, VIX, Revenue exposure to major customers | Macro Data / Supply Chain Data |

## 1.3 Pre-Calculation of Dynamic Edge Parameters

* **Rolling Correlation Parameter**:
    * Compute 30-day log returns ($r_{i}[t-30:t]$) for each stock.
    * Calculate the **Pearson correlation coefficient** ($\rho_{ij,t}$) for every stock pair.
* **Fundamental Similarity Parameter**:
    * Normalize financial metrics: P/E, P/B, ROE, Debt, Market Cap.
    * Compute the **cosine similarity** ($s_{ij}$) of the normalized vectors.