import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score
import numpy as np
import torch.nn.functional as F
import random


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.fc = nn.Linear(output_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.fc(x)


def load_condensed_graph():
    graph_data = torch.load('condensed_graph.pt')
    node_features = graph_data.x
    edge_index = graph_data.edge_index
    node_labels = graph_data.y
    return node_features, edge_index, node_labels


def load_cora_data():
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    return data


def train_and_evaluate(model, data, train_mask, test_mask, epochs=100, learning_rate=0.01):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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

    node_features, edge_index, node_labels = load_condensed_graph()

    input_dim = node_features.shape[1]
    hidden_dim = 256
    output_dim = len(torch.unique(node_labels)) 
    learning_rate = 0.001
    epochs =    1000


    node_features = node_features.to(device)
    edge_index = edge_index.to(device)
    node_labels = node_labels.to(device)

    condensed_data = Data(x=node_features, edge_index=edge_index, y=node_labels)

    model = GCN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)

    print("Training on Condensed Graph...")
    train_and_evaluate(model, condensed_data, train_mask=torch.ones_like(node_labels.bool()), test_mask=torch.ones_like(node_labels.bool()), epochs=epochs, learning_rate=learning_rate)

    print("Loading Cora dataset...")
    cora_data = load_cora_data()
    cora_data = cora_data.to(device)  

    print("Testing on Cora dataset...")
    accuracies = []
    for i in range(10):
        accuracy = train_and_evaluate(model, cora_data, train_mask=cora_data.train_mask, test_mask=cora_data.test_mask, epochs=epochs, learning_rate=learning_rate)
        accuracies.append(accuracy)
        print(f"Test {i+1} accuracy on Cora: {accuracy:.4f}")
    
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    print(f"Average test accuracy on Cora over 10 runs: {avg_accuracy:.4f}")
    print(f"Standard deviation of test accuracy: {std_accuracy:.4f}")

if __name__ == '__main__':
    main()
