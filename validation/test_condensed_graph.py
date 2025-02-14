import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from sklearn.metrics import accuracy_score
import numpy as np
import torch.nn.functional as F
import random
from models.gnn_modules import GCN
from models.load_data import load_dataset


def load_condensed_graph():
    graph_data = torch.load('condensed_graph.pt')
    return Data(x=graph_data.x, edge_index=graph_data.edge_index, y=graph_data.y)

def train_and_evaluate(model, data, train_mask, test_mask, epochs=100, lr=0.01):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        accuracy = accuracy_score(data.y[test_mask].cpu().numpy(), pred[test_mask].cpu().numpy())
    return accuracy


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = {
        'condensed': {
            'model_params': {
                'in_channels': None,  
                'out_channels': None, 
                'improved': False,
                'cached': False,
                'add_self_loops': True,
                'normalize': True,
                'bias': True
            },
            'train_params': {
                'epochs': 300,
                'lr': 0.001
            }
        },
        'cora': {
            'model_params': {
                'in_channels': None,  
                'out_channels': None, 
                'improved': False,
                'cached': False,
                'add_self_loops': True,
                'normalize': True,
                'bias': True
            },
            'train_params': {
                'epochs': 500,
                'lr': 0.001
            }
        }
    }

    condensed_data = load_condensed_graph().to(device)
    params['condensed']['model_params']['in_channels'] = condensed_data.x.size(1)
    params['condensed']['model_params']['out_channels'] = len(torch.unique(condensed_data.y))


    condensed_model = GCN(**params['condensed']['model_params']).to(device)

    print("Training on Condensed Graph...")
    condensed_acc = train_and_evaluate(
        condensed_model, condensed_data,
        train_mask=torch.ones_like(condensed_data.y.bool()),
        test_mask=torch.ones_like(condensed_data.y.bool()),
        **params['condensed']['train_params']
    )
    print(f"Condensed Graph Accuracy: {condensed_acc:.4f}")

    cora_data = load_dataset('Cora', root='data').to(device)
    params['cora']['model_params']['in_channels'] = cora_data.x.size(1)
    params['cora']['model_params']['out_channels'] = cora_data.y.unique().size(0)

    print("\nTesting on Cora Dataset:")
    accuracies = []
    for i in range(10):
        model = GCN(**params['cora']['model_params']).to(device)
        accuracy = train_and_evaluate(
            model, cora_data,
            train_mask=cora_data.train_mask,
            test_mask=cora_data.test_mask,
            **params['cora']['train_params']
        )
        accuracies.append(accuracy)
        print(f"Run {i+1}: Accuracy = {accuracy:.4f}")

    print(f"\nAverage Accuracy: {np.mean(accuracies):.4f}")
    print(f"Standard Deviation: {np.std(accuracies):.4f}")

if __name__ == '__main__':
    main()