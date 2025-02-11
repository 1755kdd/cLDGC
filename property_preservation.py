import torch
import numpy as np
from torch_geometric.datasets import Planetoid
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigsh, svds
from scipy.linalg import eigh
from sklearn.metrics import davies_bouldin_score
from scipy.stats import pearsonr

def load_original_data():
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    edge_index = data.edge_index.cpu()
    x = data.x.cpu().numpy()
    y = data.y.cpu().numpy()
    return edge_index, x, y, data.num_nodes

def get_condensed_graph():
    adj = torch.load('adj_cora_0.5.pt').cpu()
    feat = torch.load('feat_cora_0.5.pt').cpu().numpy()
    label = torch.load('label_cora_0.5.pt').cpu().numpy()
    
    if adj.is_sparse:
        adj_dense = adj.to_dense()
    else:
        adj_dense = adj.clone()
    
    adj_dense = (adj_dense + adj_dense.T) / 2 
    adj_dense.fill_diagonal_(0)  
    adj_dense = torch.where(adj_dense > 0.5, 1.0, 0.0) 
    
    edge_index = torch.stack(torch.where(adj_dense > 0), dim=0).long()
    return edge_index, feat, label, feat.shape[0]

def compute_max_eigenvalue(edge_index, num_nodes):
    rows = edge_index[0].numpy().astype(int)
    cols = edge_index[1].numpy().astype(int)
    
    if np.any(rows >= num_nodes) or np.any(cols >= num_nodes):
        print(f"error")
        print(f"max rows: {rows.max()}, max cols: {cols.max()}")
        return 0.0
    
    data = np.ones(len(rows))
    
    A = coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()
    A = A.maximum(A.T) 
    
    if A.nnz == 0:
        print("Warning")
        return 0.0
    
    try:
        eigvals_sparse = eigsh(A, k=5, which='LM', return_eigenvectors=False)
        max_eigen1 = np.max(eigvals_sparse)
        
        _, s, _ = svds(A, k=1, which='LM')
        max_singular = s[0]
        
        if num_nodes <= 1000: 
            A_dense = A.toarray()
            eigvals_dense = eigh(A_dense, eigvals_only=True, 
                                eigvals=(num_nodes-1, num_nodes-1))
            max_eigen2 = eigvals_dense[0]
            
        
            final_eigen = max(max_eigen1, max_singular, max_eigen2)
        else:
            final_eigen = max(max_eigen1, max_singular)
        
        return final_eigen.real
    except Exception as e:
        print(f"error: {str(e)}")
        return 0.0

def compute_density(edge_index, num_nodes):
    unique_edges = set()
    for u, v in edge_index.T.numpy():
        if u < v:  
            unique_edges.add((u, v))
    E = len(unique_edges)
    return (2 * E) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0.0

def compute_homophily(edge_index, labels):
    edge_set = set()
    same_count = 0
    for u, v in edge_index.T.numpy():
        if u < v:
            pair = (u, v)
            if pair not in edge_set:
                edge_set.add(pair)
                same_count += int(labels[u] == labels[v])
    return same_count / len(edge_set) if len(edge_set) > 0 else 0.0

def compute_properties(edge_index, features, labels, num_nodes):
    density = compute_density(edge_index, num_nodes)
    max_eigen = compute_max_eigenvalue(edge_index, num_nodes)
    dbi = davies_bouldin_score(features, labels)
    homophily = compute_homophily(edge_index, labels)
    return [density, max_eigen, dbi, homophily]

def main():
    edge_orig, feat_orig, label_orig, n_orig = load_original_data()
    edge_cond, feat_cond, label_cond, n_cond = get_condensed_graph()

    props_orig = compute_properties(edge_orig, feat_orig, label_orig, n_orig)
    
    props_cond = compute_properties(edge_cond, feat_cond, label_cond, n_cond)

    corr, p_value = pearsonr(props_orig, props_cond)

    print(f"\nPearson corr: {corr:.4f} (p={p_value:.4f})")

if __name__ == "__main__":
    main()