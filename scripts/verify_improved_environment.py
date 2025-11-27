# scripts/verify_improved_environment.py
"""
验证改进后的交易环境

对比原始环境和改进环境在相同条件下的表现：
1. 在上涨趋势中的仓位建立速度
2. 在下跌趋势中的风险控制
3. 整体收益和风险指标
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / 'scripts'))

from rl_environment import StockTradingEnv
from rl_environment_balanced import BalancedStockTradingEnv
from phase5_rl_integration import load_gnn_model_for_rl
from phase6_evaluation import START_DATE_TEST, END_DATE_TEST, calculate_financial_metrics


def simulate_strategy(env, num_steps=200, strategy='random'):
    """
    模拟策略执行
    
    Args:
        env: 环境对象
        num_steps: 模拟步数
        strategy: 'random' 或 'buy_all' (全部买入用于测试上涨情况)
    
    Returns:
        dict: 包含收益、仓位、交易等统计信息
    """
    obs, info = env.reset()
    
    portfolio_values = [info.get('portfolio_value', 10000)]
    positions = []  # 记录仓位变化
    trades = []  # 记录交易
    daily_returns = []
    
    initial_value = portfolio_values[0]
    
    for step in range(min(num_steps, env.max_steps)):
        # 选择动作
        if strategy == 'random':
            action = env.action_space.sample()
        elif strategy == 'buy_all':
            # 全部买入（测试上涨情况）
            action = np.array([2] * env.NUM_STOCKS)  # 全部买入
        elif strategy == 'sell_all':
            # 全部卖出（测试下跌情况）
            action = np.array([0] * env.NUM_STOCKS)  # 全部卖出
        else:
            action = env.action_space.sample()
        
        # 记录当前仓位
        try:
            # 获取当前价格
            current_date = env.data_loader['dates'][env.current_step] if env.current_step < len(env.data_loader['dates']) else None
            if current_date and current_date in env.data_loader['prices'].index:
                current_prices_row = env.data_loader['prices'].loc[current_date]
                # 提取对应ticker的价格
                ticker_prices = []
                for ticker in env.data_loader['tickers']:
                    col_name = f'Close_{ticker}'
                    if col_name in current_prices_row.index:
                        ticker_prices.append(current_prices_row[col_name])
                    else:
                        ticker_prices.append(0)
                current_prices = np.array(ticker_prices)
            else:
                current_prices = np.zeros(env.NUM_STOCKS)
            
            current_position = np.sum(env.holdings * current_prices)
            positions.append(current_position / portfolio_values[-1] if portfolio_values[-1] > 0 else 0)
        except:
            positions.append(0)
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 记录数据
        portfolio_values.append(info['portfolio_value'])
        trades.append(info.get('trades', 0))
        
        if len(portfolio_values) > 1:
            daily_return = (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
            daily_returns.append(daily_return)
        
        if terminated or truncated:
            break
    
    # 计算指标
    final_value = portfolio_values[-1]
    total_return = (final_value / initial_value - 1) * 100
    
    # 计算金融指标
    if len(portfolio_values) > 1:
        metrics = calculate_financial_metrics(portfolio_values, len(portfolio_values) - 1)
        sharpe = metrics.get('Sharpe_Ratio', 0)
        max_dd = metrics.get('Max_Drawdown', 0) * 100
    else:
        sharpe = 0
        max_dd = 0
    
    # 计算平均仓位
    avg_position = np.mean(positions) if positions else 0
    max_position = np.max(positions) if positions else 0
    
    # 计算交易次数
    total_trades = np.sum(trades)
    
    return {
        'final_value': final_value,
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'avg_position': avg_position * 100,  # 转换为百分比
        'max_position': max_position * 100,
        'total_trades': total_trades,
        'portfolio_values': portfolio_values,
        'positions': positions
    }


def compare_environments():
    """
    对比原始环境和改进环境
    """
    print("=" * 80)
    print("🔬 验证改进后的交易环境")
    print("=" * 80)
    
    # 加载 GNN 模型
    print("\n--- 加载 GNN 模型 ---")
    gnn_model = load_gnn_model_for_rl()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试场景
    test_scenarios = [
        ('random', '随机策略'),
        ('buy_all', '全部买入（测试上涨情况）'),
    ]
    
    results = []
    
    for strategy, description in test_scenarios:
        print(f"\n{'='*80}")
        print(f"📊 测试场景: {description}")
        print(f"{'='*80}")
        
        # 测试原始环境
        print("\n--- 原始环境 (固定 0.02% 买入) ---")
        env_original = StockTradingEnv(
            start_date=START_DATE_TEST,
            end_date=END_DATE_TEST,
            gnn_model=gnn_model,
            device=device
        )
        
        original_result = simulate_strategy(env_original, num_steps=200, strategy=strategy)
        
        print(f"  最终价值: ${original_result['final_value']:.2f}")
        print(f"  总收益: {original_result['total_return']:.2f}%")
        print(f"  Sharpe 比率: {original_result['sharpe_ratio']:.4f}")
        print(f"  最大回撤: {original_result['max_drawdown']:.2f}%")
        print(f"  平均仓位: {original_result['avg_position']:.2f}%")
        print(f"  最大仓位: {original_result['max_position']:.2f}%")
        print(f"  总交易次数: {original_result['total_trades']:.0f}")
        
        # 测试改进环境
        print("\n--- 改进环境 (动态仓位管理) ---")
        env_balanced = BalancedStockTradingEnv(
            start_date=START_DATE_TEST,
            end_date=END_DATE_TEST,
            gnn_model=gnn_model,
            device=device
        )
        
        balanced_result = simulate_strategy(env_balanced, num_steps=200, strategy=strategy)
        
        print(f"  最终价值: ${balanced_result['final_value']:.2f}")
        print(f"  总收益: {balanced_result['total_return']:.2f}%")
        print(f"  Sharpe 比率: {balanced_result['sharpe_ratio']:.4f}")
        print(f"  最大回撤: {balanced_result['max_drawdown']:.2f}%")
        print(f"  平均仓位: {balanced_result['avg_position']:.2f}%")
        print(f"  最大仓位: {balanced_result['max_position']:.2f}%")
        print(f"  总交易次数: {balanced_result['total_trades']:.0f}")
        
        # 计算改进
        return_improvement = balanced_result['total_return'] - original_result['total_return']
        sharpe_improvement = balanced_result['sharpe_ratio'] - original_result['sharpe_ratio']
        position_improvement = balanced_result['max_position'] - original_result['max_position']
        
        print(f"\n--- 改进效果 ---")
        print(f"  收益改进: {return_improvement:+.2f}%")
        print(f"  Sharpe 改进: {sharpe_improvement:+.4f}")
        print(f"  最大仓位改进: {position_improvement:+.2f}%")
        
        results.append({
            'scenario': description,
            'original_return': original_result['total_return'],
            'balanced_return': balanced_result['total_return'],
            'return_improvement': return_improvement,
            'original_sharpe': original_result['sharpe_ratio'],
            'balanced_sharpe': balanced_result['sharpe_ratio'],
            'sharpe_improvement': sharpe_improvement,
            'original_max_position': original_result['max_position'],
            'balanced_max_position': balanced_result['max_position'],
            'position_improvement': position_improvement,
            'original_max_dd': original_result['max_drawdown'],
            'balanced_max_dd': balanced_result['max_drawdown'],
        })
    
    # 总结
    print("\n" + "=" * 80)
    print("📊 验证结果总结")
    print("=" * 80)
    
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))
    
    # 关键发现
    print("\n" + "=" * 80)
    print("🎯 关键发现")
    print("=" * 80)
    
    avg_return_improvement = np.mean([r['return_improvement'] for r in results])
    avg_sharpe_improvement = np.mean([r['sharpe_improvement'] for r in results])
    avg_position_improvement = np.mean([r['position_improvement'] for r in results])
    
    print(f"\n1. 收益改进: 平均 {avg_return_improvement:+.2f}%")
    print(f"2. Sharpe 比率改进: 平均 {avg_sharpe_improvement:+.4f}")
    print(f"3. 仓位建立能力: 最大仓位提升 {avg_position_improvement:+.2f}%")
    
    # 验证假设
    print("\n" + "=" * 80)
    print("✅ 验证结论")
    print("=" * 80)
    
    if avg_return_improvement > 0:
        print("✅ 改进环境在收益方面表现更好")
    else:
        print("⚠️  改进环境收益需要进一步优化")
    
    if avg_position_improvement > 0:
        print("✅ 改进环境能够更快建立仓位（在上涨时更有利）")
    else:
        print("⚠️  仓位建立能力需要改进")
    
    if all(r['balanced_max_dd'] <= r['original_max_dd'] * 1.1 for r in results):
        print("✅ 改进环境保持了风险控制能力（回撤没有显著增加）")
    else:
        print("⚠️  需要检查风险控制逻辑")
    
    # 保存结果
    RESULTS_DIR = PROJECT_ROOT / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(RESULTS_DIR / 'environment_verification_results.csv', index=False)
    print(f"\n✅ 验证结果已保存到: {RESULTS_DIR / 'environment_verification_results.csv'}")
    
    return summary_df


if __name__ == '__main__':
    results_df = compare_environments()

