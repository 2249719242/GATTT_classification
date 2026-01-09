import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score
import random
import os
import sys
import argparse

# 从模型文件导入
from GATTT import ASDClassifier, compute_adj_matrix

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
    """
    按被试维度进行评估：聚合一个被试所有窗口的预测结果。
    返回: Acc, F1, AUC, Sensitivity, Specificity
    """
    model.eval()
    unique_groups = np.unique(groups)
    
    all_subject_preds = []   # 预测类别 (0或1)
    all_subject_labels = []  # 真实类别
    all_subject_probs = []   # 预测为 ASD (1) 的概率
    
    # 构造简单的 DataLoader
    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_window_probs = []
    
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            # DANN 模式下返回 (class_logits, site_logits, logits_st, logits_geo)
            outputs, _, _, _ = model(batch_x)
            probs = F.softmax(outputs, dim=1) # 转化为概率
            all_window_probs.extend(probs.cpu().numpy())
    
    all_window_probs = np.array(all_window_probs)
    
    # === 按 Subject ID (Group) 聚合结果 ===
    for sub_id in unique_groups:
        idx = np.where(groups == sub_id)[0]
        
        # Soft Voting: 平均该被试所有窗口的概率
        avg_prob = np.mean(all_window_probs[idx], axis=0) # [prob_0, prob_1]
        pred_label = np.argmax(avg_prob)
        
        all_subject_preds.append(pred_label)
        all_subject_labels.append(y[idx[0]]) # 真实标签
        all_subject_probs.append(avg_prob[1]) # 关注正类 (ASD) 的概率用于计算 AUC
        
    # === 计算各项指标 ===
    acc = accuracy_score(all_subject_labels, all_subject_preds)
    f1 = f1_score(all_subject_labels, all_subject_preds, zero_division=0)
    
    try:
        # 只有当测试集中同时包含两类样本时，才能计算 AUC
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
    # 0. 解析命令行参数
    parser = argparse.ArgumentParser(description="ASD Classification with GATTT (Ablation Studies)")
    parser.add_argument('--no_dann', action='store_true', help='Disable Domain Adversarial Training (DANN)')
    parser.add_argument('--no_geometry', action='store_true', help='Disable Tangent Space Geometry Stream')
    parser.add_argument('--no_dynamic_graph', action='store_true', help='Disable Dynamic Graph Learning')
    parser.add_argument('--output_prefix', type=str, default='5_fold_results', help='Prefix for output CSV file')
    
    args = parser.parse_args()
    
    use_dann = not args.no_dann
    use_geometry = not args.no_geometry
    use_dynamic_graph = not args.no_dynamic_graph
    
    print(f"Configurations:")
    print(f"  - DANN: {use_dann}")
    print(f"  - Geometry Stream: {use_geometry}")
    print(f"  - Dynamic Graph: {use_dynamic_graph}")
    
    # 构造输出文件名
    output_suffix = ""
    if not use_dann: output_suffix += "_no_dann"
    if not use_geometry: output_suffix += "_no_geo"
    if not use_dynamic_graph: output_suffix += "_no_dyn"
    
    csv_filename = f"{args.output_prefix}{output_suffix}.csv"
    print(f"  - Output File: {csv_filename}")
    
    # 1. 全局配置
    set_seed(42) # 固定种子
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 超参数
    BATCH_SIZE = 16 
    ACCUM_STEPS = 4 
    LR = 5e-5 
    EPOCHS = 300
    N_FOLDS = 5 # 五折交叉验证
    
    # 2. 加载数据
    print("正在加载数据...")
    if not os.path.exists('X_augmented.npy'):
        print("错误: 找不到数据文件 (X_augmented.npy 等)。请确保路径正确。")
        return

    X = np.load('X_augmented.npy')      
    y = np.load('y_augmented.npy')      
    groups = np.load('groups_augmented.npy') 
    sites = np.load('sites_augmented.npy') 
    
    num_sites = len(np.unique(sites))
    print(f"数据加载完毕. Shape: {X.shape}, Sites: {num_sites}")
    
    # 3. 准备五折交叉验证
    gkf = GroupKFold(n_splits=N_FOLDS)
    
    # 存储每一折的最佳结果
    fold_results = []
    
    print(f"\n{'='*20} 开始 {N_FOLDS}-Fold Cross Validation {'='*20}")
    
    # GroupKFold 需要由外层循环控制
    # split(X, y, groups) 返回的是索引
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"\n>>> Fold {fold+1}/{N_FOLDS}")
        print(f"训练集窗口数: {len(train_idx)} | 验证集窗口数: {len(val_idx)}")
        
        # 数据切分
        X_train, y_train, sites_train = X[train_idx], y[train_idx], sites[train_idx]
        X_val, y_val, groups_val = X[val_idx], y[val_idx], groups[val_idx]
        
        # 计算当前 Fold 训练集的邻接矩阵 (严谨做法：只用训练集计算)
        print("   计算邻接矩阵...")
        adj = compute_adj_matrix(X_train, top_k=15).to(device)
        
        # DataLoader
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.LongTensor(y_train), 
            torch.LongTensor(sites_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        
        # 初始化模型 (传入 ablation 参数)
        model = ASDClassifier(
            adj=adj, 
            num_nodes=200, 
            time_steps=80, 
            num_sites=num_sites,
            use_dann=use_dann,
            use_geometry=use_geometry,
            use_dynamic_graph=use_dynamic_graph
        ).to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-3)
        
        # 损失函数
        class_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        site_criterion = nn.CrossEntropyLoss()
        
        # 学习率调度器
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
            
            for i, (batch_x, batch_y, batch_sites) in enumerate(train_loader):
                batch_x, batch_y, batch_sites = batch_x.to(device), batch_y.to(device), batch_sites.to(device)
                
                # Dynamic Alpha for DANN
                p = float(i + epoch * len(train_loader)) / EPOCHS / len(train_loader)
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                
                # Forward
                class_logits, site_logits, logits_st, logits_geo = model(batch_x, alpha=alpha)
                
                # Loss Calculation
                loss_class = class_criterion(class_logits, batch_y)
                loss_st = class_criterion(logits_st, batch_y)
                
                # Dynamic Loss Composition
                total_loss = loss_class + 0.3 * loss_st
                
                if use_geometry and logits_geo is not None:
                    loss_geo = class_criterion(logits_geo, batch_y)
                    total_loss += 0.3 * loss_geo
                
                if use_dann and site_logits is not None:
                    loss_site = site_criterion(site_logits, batch_sites)
                    total_loss += 0.05 * loss_site
                
                # Accumulate
                loss = total_loss / ACCUM_STEPS
                loss.backward()
                
                if (i + 1) % ACCUM_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step(epoch + i / len(train_loader))
                
                running_loss += loss.item() * ACCUM_STEPS
            
            # === 验证 (按被试聚合) ===
            val_metrics = validate_subject_wise(model, X_val, y_val, groups_val, device)
            
            # 更新最佳结果 (以 Acc 为主，也可以改为 AUC 或 F1)
            if val_metrics['acc'] > best_fold_metrics['acc']:
                best_fold_metrics = val_metrics
                patience_counter = 0 # 重置计数器
                # 保存当前 Fold 的最佳模型 (可选: 加上 suffix 区分)
                # torch.save(model.state_dict(), f'best_model_fold_{fold+1}{output_suffix}.pth')
            else:
                patience_counter += 1
            
            # 打印进度
            if (epoch + 1) % 5 == 0:
                 print(f"   Epoch {epoch+1:03d} | Loss: {running_loss/len(train_loader):.4f} | "
                       f"Val Acc: {val_metrics['acc']:.4f} (Best: {best_fold_metrics['acc']:.4f}) | "
                       f"AUC: {val_metrics['auc']:.4f}")
            
            # Early Stopping Check
            if patience_counter >= patience:
                print(f"   >>> Early Stopping triggered at Epoch {epoch+1} (No improvement for {patience} epochs)")
                break
        
        print(f" Fold {fold+1} Finished. Best Metrics: {best_fold_metrics}")
        fold_results.append(best_fold_metrics)
        
    # 4. 汇总结果
    print(f"\n{'='*20} 5-Fold Result Summary ({output_suffix if output_suffix else 'Baseline'}) {'='*20}")
    metrics_keys = ['acc', 'f1', 'auc', 'sens', 'spec']
    
    # 打印平均值
    for key in metrics_keys:
        values = [res[key] for res in fold_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{key.upper()}: {mean_val:.4f} ± {std_val:.4f}")
        
    # 保存结果到 CSV
    df = pd.DataFrame(fold_results)
    # 添加 Mean 和 Std 行
    df.loc['Mean'] = df.mean()
    df.loc['Std'] = df.std()
    
    df.to_csv(csv_filename)
    print(f"\n详细结果已保存至 '{csv_filename}'")

if __name__ == "__main__":
    main()