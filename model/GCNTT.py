import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def compute_adj_matrix(data, threshold=0.1, top_k=None):
    """
    计算基于皮尔逊相关系数的邻接矩阵
    data: (Batch, Time, Nodes)
    """
    # 1. 计算所有被试的平均相关矩阵
    # 转换维度计算相关性: (Nodes, Batch * Time)
    flattened_data = data.transpose(0, 2, 1).reshape(200, -1)
    corr_matrix = np.corrcoef(flattened_data)
    
    # 2. 阈值处理
    adj = np.abs(corr_matrix) # 取绝对值，关注连接强度
    if top_k:
        # 每行只保留前 k 个最强连接
        for i in range(len(adj)):
            threshold_val = np.sort(adj[i])[-top_k]
            adj[i][adj[i] < threshold_val] = 0
    else:
        adj[adj < threshold] = 0
        
    # 3. 归一化 (D^-0.5 * A * D^-0.5)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    adj_normalized = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    
    return torch.FloatTensor(adj_normalized)


class MultiScaleSTGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleSTGCNBlock, self).__init__()
        # GCN 部分 (共享)
        self.gcn = nn.Linear(in_channels, out_channels)
        
        # 多尺度 TCN 部分 (Inception 结构)
        # 分支 1: Kernel 3
        self.tcn3 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=((3-1)//2, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.5)
        )
        # 分支 2: Kernel 5
        self.tcn5 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(5, 1), padding=((5-1)//2, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.5)
        )
        # 分支 3: Kernel 7
        self.tcn7 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(7, 1), padding=((7-1)//2, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.5)
        )
        
        # 融合层
        self.fusion = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)

    def forward(self, x, adj):
        # 1. GCN
        x = x.permute(0, 2, 3, 1)
        x = self.gcn(x)
        x = torch.matmul(x.transpose(2, 3), adj).transpose(2, 3) 
        
        # 2. Multi-Scale TCN
        x = x.permute(0, 3, 1, 2) # (B, C, T, V)
        
        b3 = self.tcn3(x)
        b5 = self.tcn5(x)
        b7 = self.tcn7(x)
        
        # 拼接并融合
        out = torch.cat([b3, b5, b7], dim=1) # (B, 3C, T, V)
        out = self.fusion(out)
        
        return out

class GradientReversal(torch.autograd.Function):
    """
    梯度反转层 (Gradient Reversal Layer)
    前向传播：直接传递输入 (Identity)
    反向传播：将梯度取反并乘以 lambda (Negative Gradient)
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)

class ASDClassifier(nn.Module):
    def __init__(self, adj, num_nodes=200, time_steps=80, hidden_dim=64, num_sites=17):
        super(ASDClassifier, self).__init__()
        
        # 回退到 Level 1: 可学习的静态图 (这也是表现最好的版本)
        self.adj = nn.Parameter(adj.clone(), requires_grad=True)
        
        # 使用多尺度 ST-GCN
        self.st_gcn1 = MultiScaleSTGCNBlock(1, 32)
        self.st_gcn2 = MultiScaleSTGCNBlock(32, hidden_dim)
        
        # 2. Transformer 阶段
        # 瓶颈层
        self.feature_bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5)
        )
        
        self.spatial_proj = nn.Linear(num_nodes * 16, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim*2, dropout=0.5, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # 3. 任务分类器 (ASD vs Control)
        # 目标：最小化 Loss
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 2)
        )
        
        # 4. 站点判别器 (Site Discriminator) -- DANN 核心
        # 目标：最大化 Loss (通过梯度反转实现)
        # 它的任务是根据提取的特征猜出是哪个站点
        self.site_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_sites) # 输出站点数量
        )

    def forward(self, x, alpha=1.0):
        x = x.unsqueeze(1)
        
        # ST-GCN
        x = self.st_gcn1(x, self.adj)
        x = self.st_gcn2(x, self.adj)
        
        # Transformer 预处理
        x = x.permute(0, 2, 3, 1) 
        x = self.feature_bottleneck(x)
        
        batch_size, time_steps, nodes, channels = x.shape
        x = x.reshape(batch_size, time_steps, -1)
        
        x = self.spatial_proj(x)
        x = self.transformer(x)
        
        # 特征提取完毕 (Common Features)
        features = torch.mean(x, dim=1)
        
        # 分支 1: ASD 诊断 (正常梯度)
        class_logits = self.classifier(features)
        
        # 分支 2: 站点判别 (反转梯度)
        # 在进入判别器之前，先把特征的梯度反转
        reversed_features = grad_reverse(features, alpha)
        site_logits = self.site_classifier(reversed_features)
        
        return class_logits, site_logits
