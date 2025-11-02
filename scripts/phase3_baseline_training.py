# phase3_baseline_training.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, HeteroConv
from torch_geometric.data import Data, HeteroData
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# FIX: Allow PyTorch to safely load torch-geometric objects (PyTorch >= 2.6)
import torch.serialization
from torch_geometric.data.storage import BaseStorage, NodeStorage, EdgeStorage, GlobalStorage
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([BaseStorage, NodeStorage, EdgeStorage, GlobalStorage])

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_GRAPHS_DIR = PROJECT_ROOT / "data" / "graphs"
MODELS_DIR = PROJECT_ROOT / "models"
OHLCV_RAW_FILE = PROJECT_ROOT / "data" / "raw" / "stock_prices_ohlcv_raw.csv"

# Hyperparameters
HIDDEN_DIM = 64
OUT_DIM = 2  # Binary classification: Up (1) or Down/Flat (0)
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
LOOKAHEAD_DAYS = 5 # 预测 5-day-ahead stock return sign [cite: 29]

# Class Imbalance Handling Configuration
# Options: 'standard' (no weighting), 'weighted' (class weights), 'focal' (focal loss)
LOSS_TYPE = 'weighted'  # Change to 'weighted' or 'focal' to handle imbalance
FOCAL_ALPHA = 0.25      # Weight for positive class in focal loss (range: 0-1)
FOCAL_GAMMA = 2.0       # Focusing parameter for focal loss (typically 2.0)

# Ensure model directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 1. Model Definition (Simple GAT Baseline) ---

class BaselineGNN(torch.nn.Module):
    """
    A simple GAT model for node-level classification.
    We will simplify and use a single edge type for the baseline.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # Using GATConv as a slightly more advanced baseline than GCN
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, dropout=0.3)
        self.lin1 = torch.nn.Linear(hidden_channels * 4, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # x: Node features [N, F_in]
        # edge_index: Adjacency list [2, E]
        
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv1(x, edge_index).relu()
        
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return x # Output: [N, 2] (logits for Up/Down)

# --- 2. Loss Functions for Class Imbalance ---

class FocalLoss(torch.nn.Module):
    """
    Focal Loss for addressing class imbalance in classification tasks.
    
    Mathematical Formulation:
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    where:
    - p_t is the model's estimated probability for the true class
    - α_t is a weighting factor for class t (balances class frequencies)
    - γ (gamma) is the focusing parameter (reduces loss for well-classified examples)
    
    Intuition:
    - When γ = 0, focal loss is equivalent to cross-entropy
    - When γ > 0, the loss focuses more on hard-to-classify examples
    - The (1 - p_t)^γ term down-weights easy examples (where p_t is high)
    - This helps the model focus on minority class and hard examples
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    Paper: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (float): Weighting factor for positive class (0-1).
                          For binary classification, alpha balances pos/neg classes.
                          Default 0.25 means negative class gets 0.75 weight.
            gamma (float): Focusing parameter. Higher values give more focus to hard examples.
                          Typical values: 2.0 (default), range [0, 5]
            reduction (str): Specifies reduction to apply: 'none' | 'mean' | 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model predictions (logits), shape [N, num_classes]
            targets: Ground truth class labels, shape [N]
        
        Returns:
            Focal loss value (scalar if reduction='mean', tensor if reduction='none')
        """
        # Step 1: Convert logits to probabilities using softmax
        # Softmax: p_i = exp(logit_i) / sum(exp(logit_j))
        p = F.softmax(inputs, dim=1)
        
        # Step 2: Create one-hot encoding of targets
        # Example: if target=1, one_hot = [0, 1] for binary classification
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Step 3: Get the probability of the true class for each example
        # p_t = p[i, true_class_i] for each example i
        p_t = p.gather(1, targets.view(-1, 1)).squeeze(1)
        
        # Step 4: Calculate focal loss components
        # (1 - p_t)^gamma: modulating factor that reduces loss for well-classified examples
        # When p_t is high (confident correct prediction), (1-p_t) is small -> loss is down-weighted
        # When p_t is low (uncertain/wrong prediction), (1-p_t) is large -> loss is emphasized
        focal_weight = (1 - p_t) ** self.gamma
        
        # Step 5: Apply alpha weighting for class balance
        # For binary: alpha for class 1, (1-alpha) for class 0
        alpha_t = torch.where(targets == 1, 
                             torch.tensor(self.alpha, device=inputs.device),
                             torch.tensor(1 - self.alpha, device=inputs.device))
        
        # Step 6: Combine all components
        # Final focal loss = alpha_t * (1-p_t)^gamma * CE_loss
        focal_loss = alpha_t * focal_weight * ce_loss
        
        # Step 7: Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss

def compute_class_weights(targets_dict, train_dates):
    """
    Computes class weights for handling imbalanced datasets.
    
    Mathematical Formula:
    w_i = n_total / (n_classes * n_i)
    
    where:
    - n_total is the total number of samples
    - n_classes is the number of classes (2 for binary)
    - n_i is the number of samples in class i
    
    Intuition:
    - Minority class gets higher weight -> more penalty when misclassified
    - Majority class gets lower weight -> less penalty when misclassified
    - This balances the contribution of each class to the loss
    
    Example:
    If we have 1000 negative samples (class 0) and 200 positive samples (class 1):
    - w_0 = 1200 / (2 * 1000) = 0.6  (majority class: lower weight)
    - w_1 = 1200 / (2 * 200) = 3.0   (minority class: higher weight)
    
    Args:
        targets_dict: Dictionary mapping dates to target tensors
        train_dates: List of training dates
    
    Returns:
        class_weights: Tensor of shape [num_classes] with weights for each class
        class_counts: Dictionary with count of each class
    """
    print("\n⚖️  Computing class weights for imbalanced data...")
    
    # Step 1: Collect all training labels
    all_train_labels = []
    for date in train_dates:
        target = targets_dict.get(date)
        if target is not None:
            all_train_labels.append(target.numpy())
    
    # Concatenate all labels into a single array
    all_train_labels = np.concatenate(all_train_labels)
    
    # Step 2: Count samples per class
    unique_classes, class_counts = np.unique(all_train_labels, return_counts=True)
    class_count_dict = dict(zip(unique_classes.astype(int), class_counts))
    
    # Step 3: Calculate total samples and number of classes
    n_total = len(all_train_labels)
    n_classes = len(unique_classes)
    
    # Step 4: Compute weights using the formula: n_total / (n_classes * n_i)
    class_weights = torch.zeros(OUT_DIM, dtype=torch.float32)
    for cls_idx in range(OUT_DIM):
        if cls_idx in class_count_dict:
            # Standard sklearn-style class weight formula
            class_weights[cls_idx] = n_total / (n_classes * class_count_dict[cls_idx])
        else:
            # If class doesn't appear in training data, assign weight of 1.0
            class_weights[cls_idx] = 1.0
    
    # Step 5: Print class distribution statistics
    print(f"   📊 Training Set Class Distribution:")
    for cls_idx in range(OUT_DIM):
        count = class_count_dict.get(cls_idx, 0)
        percentage = (count / n_total) * 100
        weight = class_weights[cls_idx].item()
        class_name = "Positive (Up)" if cls_idx == 1 else "Negative (Down/Flat)"
        print(f"      Class {cls_idx} ({class_name}): {count:6d} samples ({percentage:5.2f}%) | Weight: {weight:.4f}")
    
    # Step 6: Calculate imbalance ratio for reporting
    if len(class_count_dict) == 2:
        imbalance_ratio = max(class_counts) / min(class_counts)
        print(f"   ⚠️  Imbalance Ratio: {imbalance_ratio:.2f}:1")
        if imbalance_ratio > 2.0:
            print(f"   💡 Significant imbalance detected. Using {LOSS_TYPE} loss is recommended.")
    
    return class_weights, class_count_dict

# --- 3. Data Preparation ---

def load_graph_data(date):
    """Loads a single graph snapshot for the given date."""
    date_str = date.strftime('%Y%m%d')
    filepath = DATA_GRAPHS_DIR / f'graph_t_{date_str}.pt'
    
    if not filepath.exists():
        # This should not happen if phase 2 ran correctly with the date fix
        return None
    
    try:
        # Load the PyG graph object
        data = torch.load(filepath, weights_only=False)
        return data
    except Exception as e:
        print(f"❌ Error loading graph {filepath.name}: {e}")
        return None

def create_target_labels(tickers, dates, lookahead_days):
    """
    Calculates the 5-day ahead return sign (y_{i, t+5}) for all stocks and dates.
    [cite_start]Target: 1 if return > 0, 0 otherwise[cite: 30].
    """
    print(f"\n🏷️ Calculating {lookahead_days}-day ahead return signs...")
    
    # 1. Load OHLCV data to calculate forward returns
    ohlcv_df = pd.read_csv(OHLCV_RAW_FILE, index_col='Date', parse_dates=True)
    
    # Extract only Close prices and restructure
    close_cols = [col for col in ohlcv_df.columns if col.startswith('Close_')]
    close_prices = ohlcv_df[close_cols].copy()
    
    # 2. Calculate forward returns for all stocks
    # Shift the prices back by LOOKAHEAD_DAYS and calculate return
    # R_t+5 = (P_t+5 - P_t) / P_t
    forward_returns_df = (close_prices.shift(-lookahead_days) - close_prices) / close_prices
    
    # 3. Convert to binary labels: 1 (positive return), 0 (otherwise)
    # [cite_start]The proposal uses: 1 if > 0, 0 otherwise [cite: 30]
    target_labels = (forward_returns_df > 0).astype(int)
    
    # 4. Map targets to the graph index (Trading Days)
    targets_dict = {}
    for date in dates:
        date_str = date.strftime('%Y-%m-%d')
        if date_str in target_labels.index:
            # Extract target vector for the trading date
            target_vector = []
            for ticker in tickers:
                col_name = f'Close_{ticker}'
                if col_name in target_labels.columns:
                    target_vector.append(target_labels.loc[date_str, col_name])
                else:
                    target_vector.append(0) # Safety zero if ticker data is missing
            targets_dict[date] = torch.tensor(target_vector, dtype=torch.long)
            
    print(f"✅ Targets calculated for {len(targets_dict)} trading days.")
    return targets_dict, target_labels.index.unique() # Also return valid trading dates

# --- 3. Training and Evaluation ---

def train(model, optimizer, data, target, criterion):
    """
    Single training step for the baseline GNN.
    
    Args:
        model: The GNN model to train
        optimizer: PyTorch optimizer
        data: Graph data (HeteroData object)
        target: Ground truth labels
        criterion: Loss function (can be standard CE, weighted CE, or focal loss)
    
    Returns:
        loss_value: Scalar loss value for logging
        predictions: Predicted class labels
    """
    model.train()
    optimizer.zero_grad()
    
    # For baseline, we simplify and only use the 'rolling_correlation' edge index
    # (assuming it's the most time-dependent and dominant edge in the HeteroData)
    
    # Extract features and edge index for the BASELINE (simplified Data object logic)
    # NOTE: This assumes that the HeteroData object from Phase 2 has a 'stock' node type 
    # and a 'rolling_correlation' edge type.
    
    # We must flatten HeteroData to Data for the simple GAT/GCN baseline
    try:
        x = data['stock'].x
        # Concatenate edge_index from ALL edge types for simplicity in the baseline
        # In a real scenario, you'd select the most important edge, e.g., 'rolling_correlation'
        
        edge_index_list = []
        for metadata in data.edge_index_dict.keys():
            edge_index_list.append(data[metadata].edge_index)
        
        # If no edges, use self-loops or skip
        if not edge_index_list:
             return 0, 0
        
        # Concatenate all edge indices
        edge_index = torch.cat(edge_index_list, dim=1)
        
    except (AttributeError, KeyError) as e:
        print(f"Skipping training step: Data object structure error: {e}")
        return 0, 0
        
    out = model(x.to(DEVICE), edge_index.to(DEVICE))
    
    # Use the provided criterion (loss function) instead of hardcoded cross_entropy
    # This allows us to switch between standard CE, weighted CE, and focal loss
    loss = criterion(out, target.to(DEVICE))
    
    loss.backward()
    optimizer.step()
    
    # Return loss and predicted labels (for evaluation)
    return loss.item(), out.argmax(dim=1)

def evaluate(model, data, target):
    """Single evaluation step."""
    model.eval()
    
    # Same data extraction logic as in train()
    try:
        x = data['stock'].x
        edge_index_list = [data[metadata].edge_index for metadata in data.edge_index_dict.keys()]
        if not edge_index_list:
             return 0, 0, 0
        edge_index = torch.cat(edge_index_list, dim=1)
    except (AttributeError, KeyError):
        return 0, 0, 0
        
    with torch.no_grad():
        out = model(x.to(DEVICE), edge_index.to(DEVICE))
    
    # Convert true labels and predictions to numpy arrays
    y_true = target.cpu().numpy()
    y_pred = out.argmax(dim=1).cpu().numpy()
    
    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')
    
    return acc, f1, out

def run_training_pipeline():
    """Main function to run the time-series training loop."""
    
    # 1. Setup
    # Load the first available graph to determine dimensions and tickers
    graph_files = sorted(list(DATA_GRAPHS_DIR.glob('graph_t_*.pt')))
    if not graph_files:
        print("❌ CRITICAL: No graph files found. Run Phase 2 first.")
        return
    sample_date = pd.to_datetime(graph_files[0].stem.split('_')[-1])
    sample_graph = load_graph_data(sample_date)
    # Use robust check for HeteroData contents
    if sample_graph is None or not hasattr(sample_graph, 'node_types') or 'stock' not in sample_graph.node_types:
        print("❌ CRITICAL: Cannot load a sample graph. Phase 2 output files are invalid. STOPPING.")
        return

    # Assuming the features are stored as [N, F] tensor under 'stock'
    INPUT_DIM = sample_graph['stock'].x.shape[1]
    tickers = list(getattr(sample_graph, 'tickers', []))
    if not tickers:
        print("❌ CRITICAL: Sample graph missing tickers metadata. STOPPING.")
        return
    
    # 2. Get Targets and Date Range
    # Discover all graph dates first (files already listed above)
    graph_dates = [pd.to_datetime(f.stem.split('_')[-1]) for f in graph_files]
    # Load raw data and calculate targets on graph_dates
    targets_dict, _ = create_target_labels(tickers, graph_dates, LOOKAHEAD_DAYS)
    
    # Align training dates: intersect dates that have both a graph AND a target
    training_dates = sorted(list(set(targets_dict.keys()).intersection(graph_dates)))
    
    if not training_dates:
        print("❌ CRITICAL: No overlapping dates found between graphs and targets. STOPPING.")
        return
        
    # 3. Split Data: Simple time-based split
    TRAIN_END_DATE = training_dates[int(len(training_dates) * 0.70)]
    VAL_END_DATE = training_dates[int(len(training_dates) * 0.85)]
    
    train_dates = [d for d in training_dates if d <= TRAIN_END_DATE]
    val_dates = [d for d in training_dates if TRAIN_END_DATE < d <= VAL_END_DATE]
    test_dates = [d for d in training_dates if d > VAL_END_DATE]
    
    print(f"\n📊 Data Split (Trading Days):")
    print(f"   - Train: {len(train_dates)} days (End: {TRAIN_END_DATE.date()})")
    print(f"   - Val:   {len(val_dates)} days (End: {VAL_END_DATE.date()})")
    print(f"   - Test:  {len(test_dates)} days")
    
    # 4. Setup Loss Function (with class imbalance handling)
    print(f"\n🎯 Loss Function Configuration: {LOSS_TYPE}")
    
    if LOSS_TYPE == 'weighted':
        # Compute class weights from training data
        class_weights, class_counts = compute_class_weights(targets_dict, train_dates)
        class_weights = class_weights.to(DEVICE)
        
        # Create weighted cross-entropy loss
        # The weight parameter assigns a manual rescaling weight to each class
        # loss = -w[class] * log(p[class])
        criterion = lambda pred, target: F.cross_entropy(pred, target, weight=class_weights)
        print(f"   ✅ Using Weighted Cross-Entropy with computed class weights")
        
    elif LOSS_TYPE == 'focal':
        # Use Focal Loss to handle class imbalance
        criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
        print(f"   ✅ Using Focal Loss (alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA})")
        print(f"   💡 Focal loss automatically focuses on hard-to-classify examples")
        
    else:  # 'standard'
        # Standard cross-entropy without any class balancing
        criterion = lambda pred, target: F.cross_entropy(pred, target)
        print(f"   ⚠️  Using Standard Cross-Entropy (no class balancing)")
        print(f"   💡 Consider using 'weighted' or 'focal' if classes are imbalanced")
    
    # 5. Model Setup
    model = BaselineGNN(INPUT_DIM, HIDDEN_DIM, OUT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 6. Training Loop (Transductive/Sequential)
    print("\n🔨 Starting Sequential Training...")
    best_val_f1 = 0.0
    
    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss = 0
        
        # --- Train Phase ---
        for date in tqdm(train_dates, desc=f"Epoch {epoch} Training"):
            data = load_graph_data(date)
            target = targets_dict.get(date)
            if data and target is not None:
                loss, _ = train(model, optimizer, data, target, criterion)
                total_loss += loss
        
        avg_loss = total_loss / len(train_dates)

        # --- Validation Phase ---
        val_accs, val_f1s = [], []
        for date in val_dates:
            data = load_graph_data(date)
            target = targets_dict.get(date)
            if data and target is not None:
                acc, f1, _ = evaluate(model, data, target)
                val_accs.append(acc)
                val_f1s.append(f1)
        
        avg_val_acc = np.mean(val_accs)
        avg_val_f1 = np.mean(val_f1s)

        print(f"Epoch {epoch:02d} | Train Loss: {avg_loss:.4f} | Val Acc: {avg_val_acc:.4f} | Val F1: {avg_val_f1:.4f}")

        # Save best model based on F1 score (important metric for imbalanced finance data)
        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            torch.save(model.state_dict(), MODELS_DIR / 'baseline_gcn_model.pt')
            print(f"  --> Model Saved (New Best F1: {best_val_f1:.4f})")
    
    # 7. Testing (Final Evaluation)
    model.load_state_dict(torch.load(MODELS_DIR / 'baseline_gcn_model.pt', weights_only=False))
    test_accs, test_f1s = [], []
    
    for date in test_dates:
        data = load_graph_data(date)
        target = targets_dict.get(date)
        if data and target is not None:
            acc, f1, _ = evaluate(model, data, target)
            test_accs.append(acc)
            test_f1s.append(f1)
            
    print("\n" + "=" * 50)
    print(f"🚀 Final Test Results (Averaged over {len(test_dates)} days):")
    print(f"   - Test Accuracy: {np.mean(test_accs):.4f}")
    print(f"   - Test F1 Score: {np.mean(test_f1s):.4f}")
    print("=" * 50)


if __name__ == '__main__':
    run_training_pipeline()