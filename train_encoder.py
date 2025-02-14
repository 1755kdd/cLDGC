import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
import random
from models.load_data import load_dataset
from Configs.encoder import Config

class GraphEncoder(nn.Module):
    def __init__(self, input_dim, num_classes, dataset_name):
        super().__init__()
        
        config = Config.get_config(dataset_name)
        
        self.conv1 = GCNConv(input_dim, config['hidden_dim'])
        self.conv2 = GCNConv(config['hidden_dim'], config['output_dim'])
        
        self.edge_projector = nn.Sequential(
            nn.Linear(2 * config['output_dim'], config['edge_proj_hidden']),
            nn.ReLU(),
            nn.Linear(config['edge_proj_hidden'], config['output_dim'])
        )
        
        self.classifier = nn.Linear(config['output_dim'], num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        f = self.conv2(x, edge_index)
        
        row, col = edge_index
        f_src = f[row]
        f_dst = f[col]
        edge_feats = self.edge_projector(torch.cat([f_src, f_dst], -1))
        
        logits = self.classifier(f)
        
        return f, edge_feats, logits, edge_index


def train_and_save(dataset_name):

    config = Config.get_config(dataset_name)
    
    data = load_dataset(dataset_name, root=config['data_root']).to(config['device'])
    
    num_classes = len(torch.unique(data.y))
    
    model = GraphEncoder(
        input_dim=data.num_features,
        num_classes=num_classes,
        dataset_name=dataset_name 
    ).to(config['device'])
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    criterion = nn.CrossEntropyLoss()
    
    best_loss = float('inf')
    for epoch in range(config['epochs']):
        model.train()
        optimizer.zero_grad()
        
        _, _, logits, _ = model(data.x, data.edge_index)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])
        
        loss.backward()
        optimizer.step()
        
        if loss < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), f'best_{dataset_name}.pth')
    
    save_embeddings(model, data, dataset_name)


def save_embeddings(model, data, dataset_name):
    model.eval()
    with torch.no_grad():
        f, edge_feats, _, edge_index = model(data.x, data.edge_index)
        
        f = f.cpu()
        edge_feats = edge_feats.cpu()
        edge_index = edge_index.cpu()
        train_mask = data.train_mask.cpu()
        
        save_data = {
            'node_features': f[train_mask],
            'edge_features': edge_feats,
            'edge_index': edge_index,
            'labels': data.y[train_mask].cpu()
        }
        
        save_file = f'{dataset_name}_embeddings.pt'
        torch.save(save_data, save_file)
        print(f'[{dataset_name}] is saved as {save_file}')


if __name__ == "__main__":
    for dataset in Config.datasets:
        print(f'\n=== Processing dataset：{dataset.upper()} ===')
        train_and_save(dataset)
        torch.cuda.empty_cache()
