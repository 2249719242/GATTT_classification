import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
import random
import os
import sys

# Import GCN Only model
# Ensure GCN.py is in the same directory or PYTHONPATH
from GCN import GCNOnly
# Assumes compute_adj_matrix can be imported from GATTT or model1 as before
# If model1 is not available, we can redefine compute_adj_matrix here or import from GATTT
try:
    from GATTT import compute_adj_matrix
except ImportError:
    # Fallback definition if import fails
    def compute_adj_matrix(data, threshold=0.1, top_k=None):
        flattened_data = data.transpose(0, 2, 1).reshape(200, -1)
        corr_matrix = np.corrcoef(flattened_data)
        adj = np.abs(corr_matrix)
        if top_k:
            for i in range(len(adj)):
                threshold_val = np.sort(adj[i])[-top_k]
                adj[i][adj[i] < threshold_val] = 0
        else:
            adj[adj < threshold] = 0
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        adj_normalized = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return torch.FloatTensor(adj_normalized)

def set_seed(seed=42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def validate_subject_wise(model, X, y, groups, device, batch_size=32):
    """
    Evaluate subject-wise performance: aggregate predictions for all windows of a subject.
    Returns: Acc, F1, AUC, Sensitivity, Specificity
    """
    model.eval()
    unique_groups = np.unique(groups)
    
    all_subject_preds = []   
    all_subject_labels = [] 
    all_subject_probs = []   
    
    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_window_probs = []
    
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            # GCNOnly forward returns (logits, None)
            logits, _ = model(batch_x)
            probs = F.softmax(logits, dim=1)
            all_window_probs.extend(probs.cpu().numpy())
    
    all_window_probs = np.array(all_window_probs)
    
    # === Aggregate by Subject ID (Group) ===
    for sub_id in unique_groups:
        idx = np.where(groups == sub_id)[0]
        
        # Soft Voting
        avg_prob = np.mean(all_window_probs[idx], axis=0) 
        pred_label = np.argmax(avg_prob)
        
        all_subject_preds.append(pred_label)
        all_subject_labels.append(y[idx[0]]) 
        all_subject_probs.append(avg_prob[1]) 
        
    # === Metrics ===
    acc = accuracy_score(all_subject_labels, all_subject_preds)
    f1 = f1_score(all_subject_labels, all_subject_preds, zero_division=0)
    
    try:
        if len(np.unique(all_subject_labels)) > 1:
            auc = roc_auc_score(all_subject_labels, all_subject_probs)
        else:
            auc = 0.5 
    except ValueError:
        auc = 0.5
    
    tn, fp, fn, tp = confusion_matrix(all_subject_labels, all_subject_preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        'acc': acc, 
        'f1': f1, 
        'auc': auc, 
        'sens': sensitivity, 
        'spec': specificity
    }

def main():
    # 1. Configuration
    set_seed(42) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    BATCH_SIZE = 64 # GCN is lighter, can use larger batch size
    LR = 5e-5
    EPOCHS = 200
    N_FOLDS = 5
                           
    # 2. Load Data
    print("Loading augmented data for GCN Ablation...")
    if not os.path.exists('X_augmented.npy'):
        print("Error: Data files not found.")
        return

    X = np.load('X_augmented.npy')      
    y = np.load('y_augmented.npy')      
    groups = np.load('groups_augmented.npy') 
    
    print(f"Data Loaded. Shape: {X.shape}")
    
    # 3. 5-Fold Cross Validation
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    fold_results = []
    
    print(f"\n{'='*20} Starting GCN-Only {N_FOLDS}-Fold CV {'='*20}")
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"\n>>> Fold {fold+1}/{N_FOLDS}")
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val, groups_val = X[val_idx], y[val_idx], groups[val_idx]
        
        # Compute Adjacency Matrix for this fold's training data
        print("   Computing Adjacency Matrix...")
        adj = compute_adj_matrix(X_train, top_k=15).to(device)
        
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), 
            batch_size=BATCH_SIZE, shuffle=True, drop_last=True
        )
        
        # Initialize Model
        model = GCNOnly(adj=adj, input_dim=1, hidden_dim=64, num_classes=2).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        # Scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-6)
        
        best_fold_metrics = {'acc': 0.0, 'f1': 0.0, 'auc': 0.0, 'sens': 0.0, 'spec': 0.0}
        
        # === Training Loop ===
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            optimizer.zero_grad()
            
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                logits, _ = model(batch_x)
                loss = criterion(logits, batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                # Step scheduler per batch (approx) or per epoch
                # scheduler.step(epoch + i / len(train_loader)) # for Cosine
                
                running_loss += loss.item()
            
            # Update scheduler at epoch end for Cosine
            scheduler.step(epoch)

            # === Validation ===
            val_metrics = validate_subject_wise(model, X_val, y_val, groups_val, device)
            
            if val_metrics['acc'] > best_fold_metrics['acc']:
                best_fold_metrics = val_metrics
                # torch.save(model.state_dict(), f'best_gcn_fold_{fold+1}.pth')
            
            if (epoch + 1) % 10 == 0:
                 print(f"   Epoch {epoch+1:03d} | Loss: {running_loss/len(train_loader):.4f} | "
                       f"Val Acc: {val_metrics['acc']:.4f} (Best: {best_fold_metrics['acc']:.4f})")
        
        print(f" Fold {fold+1} Finished. Best Metrics: {best_fold_metrics}")
        fold_results.append(best_fold_metrics)
        
    # 4. Summary
    print(f"\n{'='*20} GCN-Only 5-Fold Summary {'='*20}")
    metrics_keys = ['acc', 'f1', 'auc', 'sens', 'spec']
    
    for key in metrics_keys:
        values = [res[key] for res in fold_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{key.upper()}: {mean_val:.4f} ± {std_val:.4f}")
        
    # Save results
    df = pd.DataFrame(fold_results)
    df.loc['Mean'] = df.mean()
    df.loc['Std'] = df.std()
    
    csv_filename = '5_fold_results_gcn.csv'
    df.to_csv(csv_filename)
    print(f"\nDetailed results saved to '{csv_filename}'")

if __name__ == "__main__":
    main()
