import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_scipy_sparse_matrix
import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.metrics import davies_bouldin_score

def compute_density(edge_index, num_nodes):
    if edge_index.shape[1] == 0:
        return 0.0
    edge_index = edge_index.cpu()
    edge_index_sorted = torch.sort(edge_index, dim=0).values
    edge_index_np = edge_index_sorted.T.detach().numpy()
    unique_edges = np.unique(edge_index_np, axis=0)
    num_edges = unique_edges.shape[0]
    max_possible = num_nodes * (num_nodes - 1) / 2
    return num_edges / max_possible if max_possible > 0 else 0.0

def compute_max_eigenvalue(edge_index, num_nodes):
    if edge_index.shape[1] == 0:
        return 0.0
    edge_index = edge_index.cpu()
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
    if adj.sum() == 0:
        return 0.0
    eigenvalues = eigsh(adj, k=1, which='LM', return_eigenvectors=False)
    return eigenvalues[0].real

def compute_homophily(edge_index, y):
    if edge_index.shape[1] == 0:
        return 0.0
    edge_index = edge_index.cpu()
    y = y.cpu()
    src, dst = edge_index
    mask = src < dst
    src = src[mask]
    dst = dst[mask]
    if src.shape[0] == 0:
        return 0.0
    same = (y[src] == y[dst]).sum().item()
    return same / src.shape[0]

def compute_dbi(x, y):
    if len(torch.unique(y)) == 1:
        return 0.0
    x = x.detach().cpu()
    y = y.detach().cpu()
    x_np = x.numpy() if torch.is_tensor(x) else x
    y_np = y.numpy() if torch.is_tensor(y) else y
    return davies_bouldin_score(x_np, y_np)


dataset = Planetoid(root='data/Planetoid', name='Cora')
data = dataset[0]

condensed = torch.load('condensed_graph.pt', map_location='cpu')
condensed_x = condensed['x'].detach().cpu()  # 添加detach()
condensed_edge_index = condensed['edge_index'].detach().cpu()
condensed_y = condensed['y'].detach().cpu()


original_properties = {
    'density': compute_density(data.edge_index, data.num_nodes),
    'max_eigenvalue': compute_max_eigenvalue(data.edge_index, data.num_nodes),
    'homophily': compute_homophily(data.edge_index, data.y),
    'dbi': compute_dbi(data.x, data.y),
    'dbi_agg': compute_dbi(data.x, data.y)
}

condensed_properties = {
    'density': compute_density(condensed_edge_index, condensed_x.shape[0]),
    'max_eigenvalue': compute_max_eigenvalue(condensed_edge_index, condensed_x.shape[0]),
    'homophily': compute_homophily(condensed_edge_index, condensed_y),
    'dbi': compute_dbi(condensed_x, condensed_y),
    'dbi_agg': compute_dbi(condensed_x, condensed_y)
}

print("Original Graph Properties:")
for k, v in original_properties.items():
    print(f"{k}: {v:.4f}")

print("\nCondensed Graph Properties:")
for k, v in condensed_properties.items():
    print(f"{k}: {v:.4f}")

print("\nNote: Pearson correlation requires multiple samples. Only single values are available here.")