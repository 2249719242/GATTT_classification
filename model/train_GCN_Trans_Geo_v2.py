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

# Import Components
from GCNTT import MultiScaleSTGCNBlock, compute_adj_matrix
from GATTT import TemporalAttentionPooling, grad_reverse, tangent_space_projection

# ==========================================
# GCN + Transformer + Geometry Stream V2 (Noisy & Robust)
# ==========================================
class GCNTransGeoModelV2(nn.Module):
    def __init__(self, adj, num_nodes=200, time_steps=80, num_sites=17, hidden_dim=64): # Reverted to 64
        super(GCNTransGeoModelV2, self).__init__()
        
        # --- Graph Structure ---
        self.adj = nn.Parameter(adj.clone(), requires_grad=True)
        
        # --- Stream 1: ST-GCN + Transformer ---
        self.st_gcn1 = MultiScaleSTGCNBlock(1, 32)
        self.st_gcn2 = MultiScaleSTGCNBlock(32, hidden_dim) 
        
        self.feature_bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.6) 
        )
        
        self.spatial_proj = nn.Linear(num_nodes * 32, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim*2, dropout=0.6, batch_first=True 
        )
        # Reverted Transformer Layers to 2
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.temp_pool = TemporalAttentionPooling(hidden_dim)
        
        # --- Stream 2: Geometry (Tangent Space) ---
        tangent_input_dim = num_nodes * (num_nodes + 1) // 2
        self.tangent_net = nn.Sequential(
            nn.Dropout(0.7), 
            nn.Linear(tangent_input_dim, 128), # Reverted 64 -> 128
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.6),
            nn.Linear(128, hidden_dim), 
            nn.ReLU()
        )
        
        # --- Fusion & Classification ---
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.6), 
            nn.Linear(64, 2)
        )
        
        self.site_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_sites)
        )

    def forward(self, x, alpha=1.0):
        # x: (Batch, Time, Nodes)
        
        # === Input Noise Injection (Augmentation) ===
        if self.training:
           # Add Gaussian noise with std=0.02
           x = x + torch.randn_like(x) * 0.02
           
        B, T, N = x.shape
        
        # --- 1. GCN Stream ---
        x_gcn = x.unsqueeze(1) 
        adj_expanded = self.adj 
        
        x_gcn = self.st_gcn1(x_gcn, adj_expanded)
        x_gcn = self.st_gcn2(x_gcn, adj_expanded) 
        
        x_gcn = x_gcn.permute(0, 2, 3, 1) # (B, T, N, 64)
        x_gcn = self.feature_bottleneck(x_gcn) # (B, T, N, 32)
        x_gcn = x_gcn.reshape(B, T, -1) # (B, T, N*32)
        
        x_gcn = self.spatial_proj(x_gcn) 
        x_gcn = self.transformer(x_gcn)
        feat_gcn = self.temp_pool(x_gcn) 
        
        # --- 2. Geometry Stream ---
        feat_geo_raw = tangent_space_projection(x)
        feat_geo = self.tangent_net(feat_geo_raw)
        
        # --- Fusion ---
        feat_fused = torch.cat([feat_gcn, feat_geo], dim=1)
        
        # --- Output ---
        class_logits = self.classifier(feat_fused)
        reversed_features = grad_reverse(feat_fused, alpha)
        site_logits = self.site_classifier(reversed_features)
        
        return class_logits, site_logits

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def validate_subject_wise(model, X, y, groups, device, batch_size=32):
    model.eval()
    unique_groups = np.unique(groups)
    all_subject_preds, all_subject_labels, all_subject_probs = [], [], []
    
    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_window_probs = []
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            outputs, _ = model(batch_x)
            probs = F.softmax(outputs, dim=1)
            all_window_probs.extend(probs.cpu().numpy())
    
    all_window_probs = np.array(all_window_probs)
    
    for sub_id in unique_groups:
        idx = np.where(groups == sub_id)[0]
        avg_prob = np.mean(all_window_probs[idx], axis=0)
        pred = np.argmax(avg_prob)
        all_subject_preds.append(pred)
        all_subject_labels.append(y[idx[0]])
        all_subject_probs.append(avg_prob[1])
        
    acc = accuracy_score(all_subject_labels, all_subject_preds)
    f1 = f1_score(all_subject_labels, all_subject_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_subject_labels, all_subject_probs) if len(np.unique(all_subject_labels)) > 1 else 0.5
    except ValueError: auc = 0.5
    tn, fp, fn, tp = confusion_matrix(all_subject_labels, all_subject_preds).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {'acc': acc, 'f1': f1, 'auc': auc, 'sens': sens, 'spec': spec}

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    BATCH_SIZE = 16
    ACCUM_STEPS = 4
    LR = 5e-5 # Reduced LR further
    EPOCHS = 100
    N_FOLDS = 5
    
    print("Loading data for GCN + Trans + Geo V2 (Noisy & Robust)...")
    if not os.path.exists('X_augmented.npy'): return
    X, y = np.load('X_augmented.npy'), np.load('y_augmented.npy')
    groups, sites = np.load('groups_augmented.npy'), np.load('sites_augmented.npy')
    num_sites = len(np.unique(sites))
    
    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_results = []
    
    print(f"\n{'='*20} V2 Robust Training (Noisy) {'='*20}")
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"\n>>> Fold {fold+1}/{N_FOLDS}")
        X_train, y_train, sites_train = X[train_idx], y[train_idx], sites[train_idx]
        X_val, y_val, groups_val = X[val_idx], y[val_idx], groups[val_idx]
        
        print("   Computing Adjacency Matrix...")
        adj = compute_adj_matrix(X_train, top_k=15).to(device)
        
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train), torch.LongTensor(sites_train)), 
            batch_size=BATCH_SIZE, shuffle=True, drop_last=True
        )
        
        # Reverted Hidden Dim: 32 -> 64
        model = GCNTransGeoModelV2(adj=adj, num_nodes=200, time_steps=80, num_sites=num_sites, hidden_dim=64).to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=2e-2) 
        
        class_criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
        site_criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-6)
        
        best_metrics = {'acc': 0.0}
        patience = 20
        patience_curr = 0
        
        for epoch in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
            running_loss = 0.0
            
            all_train_preds = []
            all_train_labels = []
            
            for i, (bx, by, bsites) in enumerate(train_loader):
                bx, by, bsites = bx.to(device), by.to(device), bsites.to(device)
                p = float(i + epoch * len(train_loader)) / EPOCHS / len(train_loader)
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                
                c_logits, s_logits = model(bx, alpha)
                loss = (class_criterion(c_logits, by) + 0.05 * site_criterion(s_logits, bsites)) / ACCUM_STEPS
                loss.backward()
                
                if (i+1) % ACCUM_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step(epoch + i/len(train_loader))
                running_loss += loss.item() * ACCUM_STEPS
                
                _, preds = torch.max(c_logits, 1)
                all_train_preds.extend(preds.cpu().numpy())
                all_train_labels.extend(by.cpu().numpy())

            train_acc = accuracy_score(all_train_labels, all_train_preds)
            val = validate_subject_wise(model, X_val, y_val, groups_val, device)
            
            if val['acc'] > best_metrics['acc']:
                best_metrics = val
                patience_curr = 0
            else:
                patience_curr += 1
                
            if (epoch+1) % 5 == 0:
                print(f"   Epoch {epoch+1:03d} | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val['acc']:.4f}")
            
            if patience_curr >= patience:
                print(f"   Early Stop at Ep {epoch+1}")
                break
                
        print(f" Fold {fold+1} Best: {best_metrics}")
        fold_results.append(best_metrics)

    print(f"\n{'='*20} V2 Robust Summary {'='*20}")
    for k in ['acc', 'f1', 'auc', 'sens', 'spec']:
        vals = [r.get(k, 0) for r in fold_results]
        print(f"{k.upper()}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
        
    df = pd.DataFrame(fold_results)
    df.loc['Mean'] = df.mean()
    df.loc['Std'] = df.std()
    df.to_csv('5_fold_results_gcn_trans_g o_v2.csv')
    print("Saved to 5_fold_results_gcn_trans_geo_v2.csv")

if __name__ == "__main__":
    main()
