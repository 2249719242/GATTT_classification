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

# Import Transformer Only model
from Transformer import TransformerOnly

def set_seed(seed=42):
    """固定所有随机种子以确保结果可复现"""
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
            # Transformer returns (logits, None)
            outputs, _ = model(batch_x)
            probs = F.softmax(outputs, dim=1)
            all_window_probs.extend(probs.cpu().numpy())
    
    all_window_probs = np.array(all_window_probs)
    
    # === 按 Subject ID (Group) 聚合结果 ===
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
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0 
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        'acc': acc, 
        'f1': f1, 
        'auc': auc, 
        'sens': sensitivity, 
        'spec': specificity
    }

def main():
    # 1. 全局配置
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 超参数
    BATCH_SIZE = 64 # Transformer 训练较快，可以用大 Batch
    LR = 5e-5 
    EPOCHS = 100
    N_FOLDS = 5 # 五折交叉验证
    
    # 2. 加载数据
    print("Loading augmented data for Transformer Ablation...")
    if not os.path.exists('X_augmented.npy'):
        print("错误: 找不到数据文件。")
        return

    X = np.load('X_augmented.npy')      
    y = np.load('y_augmented.npy')      
    groups = np.load('groups_augmented.npy') 
    
    print(f"Data Loaded. Shape: {X.shape}")
    
    # 3. 准备五折交叉验证
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    fold_results = []
    
    print(f"\n{'='*20} Transformer Only 5-Fold CV {'='*20}")
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"\n>>> Fold {fold+1}/{N_FOLDS}")
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val, groups_val = X[val_idx], y[val_idx], groups[val_idx]
        
        # DataLoader
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        
        # 初始化模型
        model = TransformerOnly(num_nodes=200, hidden_dim=64, num_classes=2).to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        # Scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-6)
        
        best_fold_metrics = {'acc': 0.0, 'f1': 0.0, 'auc': 0.0, 'sens': 0.0, 'spec': 0.0}
        
        # Early Stopping
        patience = 15
        patience_counter = 0

        # === 训练循环 ===
        for epoch in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
            running_loss = 0.0
            
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                # Transformer only returns (logits, None)
                logits, _ = model(batch_x)
                
                loss = criterion(logits, batch_y)
                loss.backward()
                
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                optimizer.zero_grad()
                
                # Cosine Scheduler
                # scheduler.step(epoch + i / len(train_loader))
                
                running_loss += loss.item()
            
            # Step scheduler at end of epoch
            scheduler.step(epoch)
            
            # === 验证 ===
            val_metrics = validate_subject_wise(model, X_val, y_val, groups_val, device)
            
            if val_metrics['acc'] > best_fold_metrics['acc']:
                best_fold_metrics = val_metrics
                patience_counter = 0 
                # torch.save(model.state_dict(), f'best_trans_fold_{fold+1}.pth')
            else:
                patience_counter += 1
            
            if (epoch + 1) % 5 == 0:
                 print(f"   Epoch {epoch+1:03d} | Loss: {running_loss/len(train_loader):.4f} | "
                       f"Val Acc: {val_metrics['acc']:.4f} (Best: {best_fold_metrics['acc']:.4f})")

            # Early Stopping
            if patience_counter >= patience:
                print(f"   >>> Early Stopping triggered at Epoch {epoch+1}")
                break
        
        print(f" Fold {fold+1} Finished. Best Metrics: {best_fold_metrics}")
        fold_results.append(best_fold_metrics)
        
    # 4. 汇总结果
    print(f"\n{'='*20} Transformer Only 5-Fold Summary {'='*20}")
    metrics_keys = ['acc', 'f1', 'auc', 'sens', 'spec']
    
    for key in metrics_keys:
        values = [res[key] for res in fold_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{key.upper()}: {mean_val:.4f} ± {std_val:.4f}")
        
    df = pd.DataFrame(fold_results)
    df.loc['Mean'] = df.mean()
    df.loc['Std'] = df.std()
    
    csv_filename = '5_fold_results_transformer.csv'
    df.to_csv(csv_filename)
    print(f"\nDetailed results saved to '{csv_filename}'")

if __name__ == "__main__":
    main()
