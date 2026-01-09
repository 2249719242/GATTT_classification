import torch
import torch.nn as nn

class TransformerOnly(nn.Module):
    def __init__(self, num_nodes=200, hidden_dim=64, num_classes=2):
        super(TransformerOnly, self).__init__()
        
        self.input_proj = nn.Linear(num_nodes, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim*2, dropout=0.5, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )

    def forward(self, x, alpha=1.0):
        # x: (Batch, Time, Nodes)
        # Project nodes to features
        x = self.input_proj(x) # (Batch, Time, hidden_dim)
        
        x = self.transformer(x)
        
        # Mean pooling over time
        x = torch.mean(x, dim=1) # (Batch, hidden_dim)
        
        logits = self.classifier(x)
        
        return logits, None
