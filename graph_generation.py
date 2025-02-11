import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

class CondensedGraphDecoder(nn.Module):
    def __init__(self, prototype_dim, node_dim, edge_dim, num_classes, rho):
        super(CondensedGraphDecoder, self).__init__()
        self.prototype_to_node = nn.Linear(prototype_dim, node_dim)
        self.edge_projector = nn.Sequential(
            nn.Linear(edge_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.rho = rho
        self.num_classes = num_classes
        
    def forward(self, prototypes, edge_features, edge_index, node_labels):
        node_features = self.prototype_to_node(prototypes)
        
        edge_probs = torch.sigmoid(self.edge_projector(edge_features)).squeeze()
        
        attention_scores = self.compute_attention_scores(prototypes, node_labels)
        
        adj_matrix, selected_edges = self.select_edges(
            edge_index, 
            edge_probs,
            attention_scores,
            node_labels
        )
        
        return adj_matrix, node_features, selected_edges

    def compute_attention_scores(self, prototypes, node_labels):
        label_mask = (node_labels.unsqueeze(1) == node_labels.unsqueeze(0)).float()
        similarity = F.cosine_similarity(prototypes.unsqueeze(1), prototypes.unsqueeze(0), dim=2)
        return F.softmax(similarity * label_mask, dim=-1)

    def select_edges(self, edge_index, edge_probs, attention_scores, node_labels):
        num_nodes = attention_scores.shape[0] 
        device = attention_scores.device
        
        adj_matrix = torch.zeros((num_nodes, num_nodes), device=device)
        selected_edges = []
        
        candidate_edges = []
        for idx in range(edge_index.size(1)):
            src, dst = edge_index[:, idx]
            if src >= num_nodes or dst >= num_nodes:
                continue
                
            prob_score = edge_probs[idx]
            attn_score = attention_scores[src, dst]
            label_match = float(node_labels[src] == node_labels[dst])
            
            combined_score = prob_score * attn_score * (1 + 0.5*label_match)
            candidate_edges.append((src.item(), dst.item(), combined_score.item()))
        
        candidate_edges.sort(key=lambda x: x[2], reverse=True)
        k = int(self.rho * len(candidate_edges))
        
        for i, j, score in candidate_edges[:k]:
            if adj_matrix[i, j] == 0 and score > 0:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1  
                selected_edges.append((i, j))
                
                if torch.sum(adj_matrix[i]) > 2 * self.rho * num_nodes:
                    break
                    
        print(f"Selected {len(selected_edges)} edges with min score {candidate_edges[min(k, len(candidate_edges)-1)][2]:.4f}")
        return adj_matrix, selected_edges

def prepare_condensed_graph(prototypes, embeddings, rho=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    unique_labels = sorted(prototypes.keys())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    node_features = []
    node_labels = []
    for label in unique_labels:
        proto = prototypes[label]
        node_features.append(proto)
        node_labels.extend([label_to_idx[label]] * len(proto))
        
    node_features = torch.cat(node_features, dim=0).to(device)
    node_labels = torch.tensor(node_labels, dtype=torch.long).to(device)
    
    edge_features = embeddings['edge_features'].to(device)
    edge_index = embeddings['edge_index'].to(device)
    
    max_node = node_features.size(0) - 1
    edge_index = torch.clamp(edge_index, 0, max_node)
    
    decoder = CondensedGraphDecoder(
        prototype_dim=node_features.size(1),
        node_dim=node_features.size(1),
        edge_dim=edge_features.size(1),
        num_classes=len(unique_labels),
        rho=rho
    ).to(device)
    
    adj_matrix, condensed_features, edges = decoder(
        node_features, edge_features, edge_index, node_labels
    )
    
    save_condensed_graph(condensed_features, adj_matrix, edges, node_labels)
    return adj_matrix

def save_condensed_graph(node_features, adj_matrix, edges, node_labels, file_name='condensed_graph.pt'):
    edge_index = torch.nonzero(adj_matrix, as_tuple=False).t()
    graph_data = Data(
        x=node_features,
        edge_index=edge_index,
        y=node_labels,
        num_nodes=node_features.size(0)
    )
    torch.save(graph_data, file_name)



if __name__ == "__main__":
    prototypes = torch.load('prototypes_diffused.pt')
    embeddings = torch.load('embeddings_diffused.pt')
    
    adj = prepare_condensed_graph(
        prototypes=prototypes,
        embeddings=embeddings,
        rho=0.6  
    )
    