import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import numpy as np
import random

class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.edge_projector = nn.Sequential(
            nn.Linear(2 * output_dim, 4 * output_dim),
            nn.ReLU(),
            nn.Linear(4 * output_dim, output_dim)
        )
        self.classifier = nn.Linear(output_dim, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        f = self.conv2(x, edge_index)
        
        row, col = edge_index
        f_src = f[row]
        f_dst = f[col]
        edge_feats = self.edge_projector(torch.cat([f_src, f_dst], -1))
        
        logits = self.classifier(f)
        
        return f, edge_feats, logits, edge_index

def train_and_save_embeddings(dataset_name='Cora', save_file='embeddings.pt'):
    dataset = Planetoid(root='./data', name=dataset_name)
    data = dataset[0]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GraphEncoder(
        input_dim=dataset.num_features,
        hidden_dim=256,
        output_dim=256,
        num_classes=dataset.num_classes
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    best_loss = float('inf')
    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        _, _, logits, _ = model(data.x.to(device), data.edge_index.to(device))
        loss = criterion(logits[data.train_mask.to(device)], data.y[data.train_mask].to(device))
        loss.backward()
        optimizer.step()
        
        if loss < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), f'best_model_{dataset_name}.pt')
    
    model.eval()
    with torch.no_grad():
        f, edge_feats, _, edge_index = model(data.x.to(device), data.edge_index.to(device))
        f = f.cpu()
        edge_feats = edge_feats.cpu()
        edge_index = edge_index.cpu()
        
        train_node_indices = data.train_mask.nonzero(as_tuple=False).view(-1)  # 保持在CPU
        
        is_src_in_train = torch.isin(edge_index[0], train_node_indices)
        is_dst_in_train = torch.isin(edge_index[1], train_node_indices)
        train_edge_mask = is_src_in_train & is_dst_in_train
        
        train_edge_index = edge_index[:, train_edge_mask]
        train_edge_features = edge_feats[train_edge_mask]
        
    torch.save({
        'node_features': f[train_node_indices],
        'edge_features': train_edge_features,
        'edge_index': train_edge_index,
        'labels': data.y[train_node_indices] 
    }, save_file)
    print(f"Embeddings saved to {save_file}")


if __name__ == "__main__":
    train_and_save_embeddings()
