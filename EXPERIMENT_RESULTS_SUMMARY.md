# 

: 2025-12-04

## 1. Baseline Model Comparison

### 
- **LSTM**: Accuracy 52.05%, F1 34.68%, Precision 54.01%
- **MLP**: Accuracy 52.02%, F1 36.82%, Precision 51.28%
- **Logistic Regression**: Accuracy 50.48%, F1 50.41%, Precision 50.41%

### Graph-based Models
- **GAT**: Accuracy 49.99%, F1 49.90%, Precision 50.24%
- **GCN**: Accuracy 50.22%, F1 49.57%, Precision 49.84%
- **GraphSAGE**: Accuracy 48.35%, F1 47.97%, Precision 48.72%
- **HGT**: Accuracy 50.30%, F1 47.67%, Precision 49.38%

****: LSTM, MLP

## 2. Enhanced Ablation Study

### 
- **Full_Model**: Accuracy 54.62%, F1 35.33%, Precision@Top-10 52.16%
- **Abl_NoCorrelation**: Accuracy 54.62%, F1 35.33%, Precision@Top-10 52.16%
- **Abl_NoFundSim**: Accuracy 54.62%, F1 35.33%, Precision@Top-10 52.16%
- **Abl_NoStatic**: Accuracy 54.62%, F1 35.33%, Precision@Top-10 52.16%
- **Abl_OnlyFundSim**: Accuracy 54.62%, F1 35.33%, Precision@Top-10 52.33%
- **Abl_OnlyCorrelation**: : 'stock'

****: 
- 
- Precision@Top-10: 52.33%
- 

## 3. Lookahead Horizon Analysis

### 

| Lookahead Days | Test Accuracy | Test F1 | Test Precision | Best Val F1 |
|----------------|---------------|---------|----------------|-------------|
| 1              | 53.07%        | 34.67%  | 26.54%         | 33.60%      |
| 3              | 54.01%        | 35.07%  | 27.01%         | 33.98%      |
| 5              | 54.62%        | 35.33%  | 27.31%         | 34.21%      |
| 7              | 55.44%        | 35.67%  | 27.72%         | 34.11%      |
| 10             | 55.66%        | 35.76%  | 27.83%         | 34.24%      |

****: 
- 7-10F1
- 10Accuracy: 55.66%, F1: 35.76%
- F1534.21%

## 4. Graph Sparsification Results

: `results/graph_sparsification_results.csv`

## 5. Robustness Checks

: `results/robustness_checks_results.csv`

## 6. MARL Ablation

: `results/marl_ablation_template.json`

## 

### 
1. ****: 
2. ****: 7-10
3. ****: 
4. ****: 

### 
1.  `Abl_OnlyCorrelation` 
2. 
3. 7-10

