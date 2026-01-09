import torch
import torch.nn as nn
from GCNTT import MultiScaleSTGCNBlock

class GCNOnly(nn.Module):
    def __init__(self, adj, input_dim=1, hidden_dim=64, num_classes=2):
        super(GCNOnly, self).__init__()
        
        # 静态图结构
        self.adj = nn.Parameter(adj.clone(), requires_grad=True)
        
        self.st_gcn1 = MultiScaleSTGCNBlock(input_dim, 32)
        self.st_gcn2 = MultiScaleSTGCNBlock(32, hidden_dim)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )

    def forward(self, x, alpha=1.0):
        # x: (Batch, Time, Nodes)
        x = x.unsqueeze(1) # (B, 1, T, N)
        
        x = self.st_gcn1(x, self.adj)
        x = self.st_gcn2(x, self.adj)
        
        # x: (B, hidden_dim, T, N)
        # Global Pooling strategy: Mean over Nodes and Time
        x = torch.mean(x, dim=3) # Mean over Nodes -> (B, hidden_dim, T)
        x = torch.mean(x, dim=2) # Mean over Time -> (B, hidden_dim)
        
        logits = self.classifier(x)
        
        #为了兼容 train.py 的接口 (returning class_logits, site_logits)
        # 这里只有分类 logits, site_logits 返回 None 或者 只是为了占位
        return logits, None
