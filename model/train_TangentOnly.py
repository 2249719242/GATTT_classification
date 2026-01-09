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

# 导入必要的工具函数
from GATTT import tangent_space_projection, grad_reverse

class TangentOnlyModel(nn.Module):
    def __init__(self, num_nodes=200, hidden_dim=64, num_sites=17):
        super(TangentOnlyModel, self).__init__()
        
        # 1. 几何特征提取流
        # 维度计算: 上三角矩阵元素个数 = N * (N+1) / 2
        tangent_input_dim = num_nodes * (num_nodes + 1) // 2
        
        self.tangent_net = nn.Sequential(
            nn.Dropout(0.6), 
            nn.Linear(tangent_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.6),
            nn.Linear(128, hidden_dim), 
            nn.ReLU()
        )
        
        # 2. 任务分类器
        # 输入维度: hidden_dim (没有 ST 流的拼接)
        # 注意：这里我们增加了分类器的容量，使其与 Full Model 的分类头参数量大致相当，保证公平
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64), # 原来是 fusion_dim (hidden*2) -> 64
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        )
        
        # 3. 站点判别器 (DANN)
        self.site_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_sites)
        )

    def forward(self, x, alpha=1.0):
        # x: (B, T, N)
        
        # 1. 提取几何特征 (无参变换)
        feat_geo_raw = tangent_space_projection(x) 
        
        # 2. MLP 特征映射
        features = self.tangent_net(feat_geo_raw) 
        
        # 3. 分类
        class_logits = self.classifier(features)
        
        # 4. 站点判别
        reversed_features = grad_reverse(features, alpha)
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
    all_subject_preds = []   
    all_subject_labels = [] 
    all_subject_probs = []   
    
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
        pred_label = np.argmax(avg_prob)
        all_subject_preds.append(pred_label)
        all_subject_labels.append(y[idx[0]]) 
        all_subject_probs.append(avg_prob[1]) 
        
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
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0 
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {'acc': acc, 'f1': f1, 'auc': auc, 'sens': sensitivity, 'spec': specificity}

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    BATCH_SIZE = 64 # Geometry only is very fast, can use larger batch
    ACCUM_STEPS = 1 # No need for accumulation with larger batch
    LR = 1e-4       # Can potentially use larger LR
    EPOCHS = 100
    N_FOLDS = 5
    
    print("Loading data for Tangent-Only Baseline...")
    if not os.path.exists('X_augmented.npy'): return

    X = np.load('X_augmented.npy')      
    y = np.load('y_augmented.npy')      
    groups = np.load('groups_augmented.npy') 
    sites = np.load('sites_augmented.npy') 
    num_sites = len(np.unique(sites))
    
    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_results = []
    
    print(f"\n{'='*20} Tangent Space Only 5-Fold CV {'='*20}")
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"\n>>> Fold {fold+1}/{N_FOLDS}")
        
        X_train, y_train, sites_train = X[train_idx], y[train_idx], sites[train_idx]
        X_val, y_val, groups_val = X[val_idx], y[val_idx], groups[val_idx]
        
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train), torch.LongTensor(sites_train)), 
            batch_size=BATCH_SIZE, shuffle=True, drop_last=True
        )
        
        model = TangentOnlyModel(num_nodes=200, hidden_dim=64, num_sites=num_sites).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
        class_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        site_criterion = nn.CrossEntropyLoss()
        
        best_fold_metrics = {'acc': 0.0}
        patience = 15
        patience_counter = 0

        for epoch in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
            running_loss = 0.0
            
            for batch_x, batch_y, batch_sites in train_loader:
                batch_x, batch_y, batch_sites = batch_x.to(device), batch_y.to(device), batch_sites.to(device)
                
                # Dynamic Alpha
                p = float(epoch) / EPOCHS
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                
                class_logits, site_logits = model(batch_x, alpha)
                loss_class = class_criterion(class_logits, batch_y)
                loss_site = site_criterion(site_logits, batch_sites)
                
                loss = loss_class + 0.05 * loss_site
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                running_loss += loss.item()
            
            val_metrics = validate_subject_wise(model, X_val, y_val, groups_val, device)
            
            if val_metrics['acc'] > best_fold_metrics['acc']:
                best_fold_metrics = val_metrics
                patience_counter = 0
            else:
                patience_counter += 1
                
            if (epoch + 1) % 5 == 0:
                 print(f"   Epoch {epoch+1:03d} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_metrics['acc']:.4f}")
            
            if patience_counter >= patience:
                print(f"   >>> Early Stopping at Epoch {epoch+1}")
                break
        
        print(f" Fold {fold+1} Finished. Best Metrics: {best_fold_metrics}")
        fold_results.append(best_fold_metrics)
        
    # Result Summary (Same format)
    print(f"\n{'='*20} Tangent Only Summary {'='*20}")
    metrics_keys = ['acc', 'f1', 'auc', 'sens', 'spec']
    for key in metrics_keys:
        values = [res.get(key, 0) for res in fold_results]
        print(f"{key.upper()}: {np.mean(values):.4f} ± {np.std(values):.4f}")
        
    df = pd.DataFrame(fold_results)
    df.loc['Mean'] = df.mean()
    df.loc['Std'] = df.std()
    df.to_csv('5_fold_results_tangent_only.csv')
    print(f"\nDetailed results saved.")

if __name__ == "__main__":
    main()
