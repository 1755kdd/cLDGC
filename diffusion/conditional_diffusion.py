import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion.ldm import ConditionalDiffusion

class ConditionalLatentDiffusion(ConditionalDiffusion):
    def __init__(self, feat_dim, label_dim, hidden_dim=256, timesteps=1000):
        super().__init__(feat_dim, label_dim, hidden_dim, timesteps)
        
        self.label_emb = nn.Embedding(label_dim, hidden_dim)

        # Cross-Attention Layer
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4)

    def forward(self, x, h, y, t):
        y_emb = self.label_emb(y)
        
        x_in = torch.cat([x, y_emb], dim=-1)
        h_in = torch.cat([h, y_emb], dim=-1)
        
        node_out = self.node_net(x_in)
        edge_out = self.edge_net(h_in)
        
        return node_out, edge_out

    def diffuse(self, x, h, y):
        return super().diffuse(x, h, y)
