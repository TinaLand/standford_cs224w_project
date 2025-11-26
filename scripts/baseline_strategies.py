# scripts/baseline_strategies.py
"""
Baseline Trading Strategies for Comparison

Implements simple trading strategies to compare against the RL agent:
1. Buy-and-Hold: Buy all stocks at the start and hold
2. Equal-Weight: Rebalance to equal weights daily
3. Market Index (S&P 500 proxy): Equal-weight portfolio of all stocks
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / 'scripts'))

from phase6_evaluation import calculate_financial_metrics
from phase4_core_training import _read_time_series_csv, OHLCV_RAW_FILE


def buy_and_hold_strategy(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    initial_cash: float = 10000.0,
    transaction_cost: float = 0.001
) -> Dict:
    """
    Buy-and-Hold Strategy: Buy all stocks at the start and hold until the end.
    
    Args:
        start_date: Start date for backtesting
        end_date: End date for backtesting
        initial_cash: Initial capital
        transaction_cost: Transaction cost per trade (0.1% = 0.001)
    
    Returns:
        Dictionary with portfolio values and metrics
    """
    print("\n--- Buy-and-Hold Strategy ---")
    
    # Load price data
    ohlcv_df = _read_time_series_csv(OHLCV_RAW_FILE)
    close_cols = [col for col in ohlcv_df.columns if col.startswith('Close_')]
    tickers = [col.replace('Close_', '') for col in close_cols]
    
    # Filter dates
    price_data = ohlcv_df[close_cols].loc[start_date:end_date].copy()
    
    if price_data.empty:
        return {'portfolio_values': [initial_cash], 'final_value': initial_cash}
    
    # Get first and last prices
    first_prices = price_data.iloc[0].values
    last_prices = price_data.iloc[-1].values
    
    # Calculate initial allocation (equal weight)
    num_stocks = len(tickers)
    weight_per_stock = 1.0 / num_stocks
    cash_per_stock = initial_cash * weight_per_stock
    
    # Calculate shares bought (accounting for transaction costs)
    shares = []
    total_cost = 0
    for i, price in enumerate(first_prices):
        if pd.notna(price) and price > 0:
            shares_bought = cash_per_stock / (price * (1 + transaction_cost))
            shares.append(shares_bought)
            total_cost += cash_per_stock
        else:
            shares.append(0)
    
    # Calculate portfolio value over time
    portfolio_values = []
    for date_idx in range(len(price_data)):
        current_prices = price_data.iloc[date_idx].values
        portfolio_value = sum(shares[i] * price if pd.notna(price) and price > 0 else 0 
                             for i, price in enumerate(current_prices))
        portfolio_values.append(portfolio_value)
    
    final_value = portfolio_values[-1]
    total_days = len(portfolio_values) - 1
    
    # Calculate metrics
    metrics = calculate_financial_metrics(portfolio_values, total_days)
    
    print(f"  Initial Value: ${initial_cash:.2f}")
    print(f"  Final Value: ${final_value:.2f}")
    print(f"  Cumulative Return: {metrics['Cumulative_Return']*100:.2f}%")
    print(f"  Sharpe Ratio: {metrics['Sharpe_Ratio']:.4f}")
    print(f"  Max Drawdown: {metrics['Max_Drawdown']*100:.2f}%")
    
    return {
        'strategy': 'Buy-and-Hold',
        'portfolio_values': portfolio_values,
        'final_value': final_value,
        'total_days': total_days,
        **metrics
    }


def equal_weight_strategy(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    initial_cash: float = 10000.0,
    transaction_cost: float = 0.001,
    rebalance_frequency: str = 'daily'  # 'daily' or 'weekly'
) -> Dict:
    """
    Equal-Weight Strategy: Rebalance portfolio to equal weights periodically.
    
    Args:
        start_date: Start date for backtesting
        end_date: End date for backtesting
        initial_cash: Initial capital
        transaction_cost: Transaction cost per trade (0.1% = 0.001)
        rebalance_frequency: 'daily' or 'weekly'
    
    Returns:
        Dictionary with portfolio values and metrics
    """
    print(f"\n--- Equal-Weight Strategy (Rebalance: {rebalance_frequency}) ---")
    
    # Load price data
    ohlcv_df = _read_time_series_csv(OHLCV_RAW_FILE)
    close_cols = [col for col in ohlcv_df.columns if col.startswith('Close_')]
    tickers = [col.replace('Close_', '') for col in close_cols]
    
    # Filter dates
    price_data = ohlcv_df[close_cols].loc[start_date:end_date].copy()
    
    if price_data.empty:
        return {'portfolio_values': [initial_cash], 'final_value': initial_cash}
    
    num_stocks = len(tickers)
    portfolio_values = [initial_cash]
    cash = initial_cash
    shares = np.zeros(num_stocks)
    
    # Determine rebalance days
    if rebalance_frequency == 'daily':
        rebalance_days = list(range(len(price_data)))
    elif rebalance_frequency == 'weekly':
        rebalance_days = list(range(0, len(price_data), 5))  # Every 5 trading days
    else:
        rebalance_days = list(range(len(price_data)))
    
    for day_idx in range(len(price_data)):
        current_prices = price_data.iloc[day_idx].values
        
        # Calculate current portfolio value
        current_value = sum(shares[i] * price if pd.notna(price) and price > 0 else 0 
                           for i, price in enumerate(current_prices))
        current_value += cash
        
        portfolio_values.append(current_value)
        
        # Rebalance if needed
        if day_idx in rebalance_days and day_idx < len(price_data) - 1:
            # Calculate target allocation (equal weight)
            target_weight = 1.0 / num_stocks
            target_value_per_stock = current_value * target_weight
            
            # Rebalance each stock
            new_shares = np.zeros(num_stocks)
            total_cost = 0
            
            for i, price in enumerate(current_prices):
                if pd.notna(price) and price > 0:
                    # Target shares
                    target_shares = target_value_per_stock / price
                    
                    # Current shares value
                    current_shares_value = shares[i] * price
                    
                    # Trade needed
                    trade_value = target_value_per_stock - current_shares_value
                    
                    if abs(trade_value) > 0.01:  # Only trade if significant
                        # Apply transaction cost
                        if trade_value > 0:  # Buying
                            cost = trade_value * transaction_cost
                            actual_trade = trade_value - cost
                            new_shares[i] = shares[i] + (actual_trade / price)
                            total_cost += cost
                        else:  # Selling
                            cost = abs(trade_value) * transaction_cost
                            actual_trade = trade_value + cost
                            new_shares[i] = shares[i] + (actual_trade / price)
                            total_cost += cost
                    else:
                        new_shares[i] = shares[i]
                else:
                    new_shares[i] = 0
            
            # Update shares and cash
            shares = new_shares
            cash = current_value - sum(shares[i] * price if pd.notna(price) and price > 0 else 0 
                                      for i, price in enumerate(current_prices))
    
    final_value = portfolio_values[-1]
    total_days = len(portfolio_values) - 1
    
    # Calculate metrics
    metrics = calculate_financial_metrics(portfolio_values, total_days)
    
    print(f"  Initial Value: ${initial_cash:.2f}")
    print(f"  Final Value: ${final_value:.2f}")
    print(f"  Cumulative Return: {metrics['Cumulative_Return']*100:.2f}%")
    print(f"  Sharpe Ratio: {metrics['Sharpe_Ratio']:.4f}")
    print(f"  Max Drawdown: {metrics['Max_Drawdown']*100:.2f}%")
    
    return {
        'strategy': f'Equal-Weight ({rebalance_frequency})',
        'portfolio_values': portfolio_values,
        'final_value': final_value,
        'total_days': total_days,
        **metrics
    }


def run_all_baseline_strategies(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    initial_cash: float = 10000.0
) -> pd.DataFrame:
    """
    Run all baseline strategies and return comparison DataFrame.
    
    Args:
        start_date: Start date for backtesting
        end_date: End date for backtesting
        initial_cash: Initial capital
    
    Returns:
        DataFrame with results for all strategies
    """
    print("=" * 60)
    print("📊 Running Baseline Strategies Comparison")
    print("=" * 60)
    
    results = []
    
    # 1. Buy-and-Hold
    bh_result = buy_and_hold_strategy(start_date, end_date, initial_cash)
    results.append(bh_result)
    
    # 2. Equal-Weight (Daily Rebalance)
    ew_daily_result = equal_weight_strategy(start_date, end_date, initial_cash, rebalance_frequency='daily')
    results.append(ew_daily_result)
    
    # 3. Equal-Weight (Weekly Rebalance)
    ew_weekly_result = equal_weight_strategy(start_date, end_date, initial_cash, rebalance_frequency='weekly')
    results.append(ew_weekly_result)
    
    # Create DataFrame
    comparison_df = pd.DataFrame(results)
    
    # Select key columns
    display_cols = ['strategy', 'final_value', 'Cumulative_Return', 'Sharpe_Ratio', 'Max_Drawdown']
    comparison_df = comparison_df[display_cols]
    
    print("\n" + "=" * 60)
    print("📊 BASELINE STRATEGIES COMPARISON")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    
    return comparison_df


def main():
    """Main function to run baseline strategies."""
    from phase6_evaluation import START_DATE_TEST, END_DATE_TEST
    
    # Use same test period as RL evaluation
    start_date = START_DATE_TEST
    end_date = END_DATE_TEST
    
    print(f"Backtesting Period: {start_date.date()} to {end_date.date()}")
    
    # Run all strategies
    comparison_df = run_all_baseline_strategies(start_date, end_date, initial_cash=10000.0)
    
    # Save results
    RESULTS_DIR = PROJECT_ROOT / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(RESULTS_DIR / 'baseline_strategies_comparison.csv', index=False)
    
    print(f"\n✅ Results saved to: {RESULTS_DIR / 'baseline_strategies_comparison.csv'}")
    
    return comparison_df


if __name__ == '__main__':
    main()

