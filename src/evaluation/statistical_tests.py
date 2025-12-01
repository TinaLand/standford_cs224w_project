"""
Statistical Significance Testing for Model Evaluation

Implements various statistical tests to validate model performance improvements.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from scipy import stats
from sklearn.utils import resample


def block_bootstrap_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    block_size: int = 20,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Calculate Sharpe ratio with block bootstrap confidence intervals.
    
    Args:
        returns: Array of portfolio returns
        risk_free_rate: Annual risk-free rate (default: 2%)
        block_size: Block size for bootstrap (default: 20 days)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for CI (default: 95%)
    
    Returns:
        Dictionary with Sharpe ratio, CI lower, CI upper, p-value
    """
    if len(returns) < block_size:
        return {
            'sharpe_ratio': 0.0,
            'ci_lower': 0.0,
            'ci_upper': 0.0,
            'p_value': 1.0
        }
    
    # Annualize returns (assuming daily returns, 252 trading days)
    annualized_mean = np.mean(returns) * 252
    annualized_std = np.std(returns) * np.sqrt(252)
    annualized_rf = risk_free_rate
    
    if annualized_std < 1e-8:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = (annualized_mean - annualized_rf) / annualized_std
    
    # Block bootstrap
    n_blocks = len(returns) // block_size
    sharpe_bootstrap = []
    
    for _ in range(n_bootstrap):
        # Sample blocks with replacement
        block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)
        bootstrap_returns = []
        
        for idx in block_indices:
            start = idx * block_size
            end = min(start + block_size, len(returns))
            bootstrap_returns.extend(returns[start:end])
        
        bootstrap_returns = np.array(bootstrap_returns)
        if len(bootstrap_returns) > 0:
            b_mean = np.mean(bootstrap_returns) * 252
            b_std = np.std(bootstrap_returns) * np.sqrt(252)
            if b_std > 1e-8:
                b_sharpe = (b_mean - annualized_rf) / b_std
                sharpe_bootstrap.append(b_sharpe)
    
    if len(sharpe_bootstrap) == 0:
        return {
            'sharpe_ratio': sharpe_ratio,
            'ci_lower': sharpe_ratio,
            'ci_upper': sharpe_ratio,
            'p_value': 1.0
        }
    
    sharpe_bootstrap = np.array(sharpe_bootstrap)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(sharpe_bootstrap, 100 * alpha / 2)
    ci_upper = np.percentile(sharpe_bootstrap, 100 * (1 - alpha / 2))
    
    # P-value: probability that Sharpe ratio <= 0
    p_value = np.mean(sharpe_bootstrap <= 0)
    
    return {
        'sharpe_ratio': sharpe_ratio,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value,
        'n_bootstrap': n_bootstrap
    }


def t_test_accuracy(
    predictions1: np.ndarray,
    targets1: np.ndarray,
    predictions2: np.ndarray,
    targets2: np.ndarray
) -> Dict[str, float]:
    """
    Perform t-test to compare accuracy of two models.
    
    Args:
        predictions1: Predictions from model 1
        targets1: True labels for model 1
        predictions2: Predictions from model 2
        targets2: True labels for model 2
    
    Returns:
        Dictionary with t-statistic, p-value, and effect size
    """
    # Calculate accuracies per sample
    correct1 = (predictions1 == targets1).astype(float)
    correct2 = (predictions2 == targets2).astype(float)
    
    # Perform paired t-test (if same test set) or independent t-test
    if len(correct1) == len(correct2):
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(correct1, correct2)
    else:
        # Independent t-test
        t_stat, p_value = stats.ttest_ind(correct1, correct2)
    
    # Effect size (Cohen's d)
    mean_diff = np.mean(correct1) - np.mean(correct2)
    pooled_std = np.sqrt((np.var(correct1) + np.var(correct2)) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0
    
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'mean_accuracy_1': float(np.mean(correct1)),
        'mean_accuracy_2': float(np.mean(correct2)),
        'cohens_d': float(cohens_d),
        'significant': p_value < 0.05
    }


def wilcoxon_signed_rank_test(
    scores1: np.ndarray,
    scores2: np.ndarray
) -> Dict[str, float]:
    """
    Perform Wilcoxon signed-rank test (non-parametric alternative to t-test).
    
    Args:
        scores1: Scores from model 1
        scores2: Scores from model 2
    
    Returns:
        Dictionary with statistic, p-value
    """
    if len(scores1) != len(scores2):
        raise ValueError("Scores must have the same length for paired test")
    
    statistic, p_value = stats.wilcoxon(scores1, scores2, alternative='two-sided')
    
    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': p_value < 0.05
    }


def compare_models_statistical(
    model1_metrics: Dict[str, float],
    model2_metrics: Dict[str, float],
    model1_predictions: np.ndarray = None,
    model1_targets: np.ndarray = None,
    model2_predictions: np.ndarray = None,
    model2_targets: np.ndarray = None
) -> Dict[str, any]:
    """
    Comprehensive statistical comparison between two models.
    
    Args:
        model1_metrics: Dictionary of metrics for model 1
        model2_metrics: Dictionary of metrics for model 2
        model1_predictions: Predictions from model 1 (optional)
        model1_targets: True labels for model 1 (optional)
        model2_predictions: Predictions from model 2 (optional)
        model2_targets: True labels for model 2 (optional)
    
    Returns:
        Dictionary with statistical test results
    """
    results = {
        'model1_metrics': model1_metrics,
        'model2_metrics': model2_metrics,
        'comparisons': {}
    }
    
    # Compare accuracy if predictions available
    if (model1_predictions is not None and model1_targets is not None and
        model2_predictions is not None and model2_targets is not None):
        accuracy_test = t_test_accuracy(
            model1_predictions, model1_targets,
            model2_predictions, model2_targets
        )
        results['comparisons']['accuracy'] = accuracy_test
    
    # Compare F1 scores using Wilcoxon test (if we have per-sample scores)
    if 'f1_score' in model1_metrics and 'f1_score' in model2_metrics:
        # For aggregated metrics, we can't do per-sample test
        # But we can report the difference
        f1_diff = model1_metrics['f1_score'] - model2_metrics['f1_score']
        results['comparisons']['f1_difference'] = {
            'difference': f1_diff,
            'model1_f1': model1_metrics['f1_score'],
            'model2_f1': model2_metrics['f1_score']
        }
    
    # Compare Sharpe ratios if available
    if 'sharpe_ratio' in model1_metrics and 'sharpe_ratio' in model2_metrics:
        sharpe_diff = model1_metrics['sharpe_ratio'] - model2_metrics['sharpe_ratio']
        results['comparisons']['sharpe_difference'] = {
            'difference': sharpe_diff,
            'model1_sharpe': model1_metrics['sharpe_ratio'],
            'model2_sharpe': model2_metrics['sharpe_ratio']
        }
    
    return results


def print_statistical_summary(results: Dict[str, any]):
    """
    Print a formatted summary of statistical test results.
    """
    print("\n" + "=" * 60)
    print("📊 Statistical Test Results")
    print("=" * 60)
    
    print("\nModel 1 Metrics:")
    for key, value in results['model1_metrics'].items():
        print(f"  {key}: {value:.4f}")
    
    print("\nModel 2 Metrics:")
    for key, value in results['model2_metrics'].items():
        print(f"  {key}: {value:.4f}")
    
    print("\nComparisons:")
    for test_name, test_result in results['comparisons'].items():
        print(f"\n  {test_name}:")
        if isinstance(test_result, dict):
            for key, value in test_result.items():
                if isinstance(value, bool):
                    print(f"    {key}: {value}")
                else:
                    print(f"    {key}: {value:.4f}")
    
    print("\n" + "=" * 60)

