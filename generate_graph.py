import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

class CondensedGraphDecoder(nn.Module):
    def __init__(self, prototype_dim, node_dim, edge_dim, num_classes, rho):
        """
        Initialize the Condensed Graph Decoder.
        Args:
            prototype_dim (int): The dimension of the prototype embeddings.
            node_dim (int): The dimension of the node features.
            edge_dim (int): The dimension of the edge features.
            num_classes (int): The number of distinct classes in the graph.
            rho (float): A parameter controlling edge selection density.
        """
        super(CondensedGraphDecoder, self).__init__()
        
        # Projection layer for converting prototypes to node features
        self.prototype_to_node = nn.Linear(prototype_dim, node_dim)
        
        # Edge feature projector to compute edge probabilities
        self.edge_projector = nn.Sequential(
            nn.Linear(edge_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Density parameter for edge selection
        self.rho = rho
        # Total number of node classes
        self.num_classes = num_classes
        
    def forward(self, prototypes, edge_features, edge_index, node_labels):
        """
        Forward pass through the model, generating condensed graph components.
        
        Args:
            prototypes (Tensor): Node prototype embeddings.
            edge_features (Tensor): Features associated with the edges.
            edge_index (Tensor): Indices of the edges.
            node_labels (Tensor): Node class labels.
        
        Returns:
            adj_matrix (Tensor): The adjacency matrix of the condensed graph.
            node_features (Tensor): The decoded node features.
            selected_edges (List): The edges selected based on scores.
        """
        # Decode node features from prototype embeddings
        node_features = self.prototype_to_node(prototypes)
        
        # Calculate edge probabilities
        edge_probs = torch.sigmoid(self.edge_projector(edge_features)).squeeze()
        
        # Compute attention scores based on node labels and prototype similarities
        attention_scores = self.compute_attention_scores(prototypes, node_labels)
        
        # Select edges based on computed scores and edge probabilities
        adj_matrix, selected_edges = self.select_edges(
            edge_index, 
            edge_probs,
            attention_scores,
            node_labels
        )
        
        return adj_matrix, node_features, selected_edges

    def compute_attention_scores(self, prototypes, node_labels):
        """
        Compute attention scores based on prototype similarity and node labels.
        
        Args:
            prototypes (Tensor): Node prototype embeddings.
            node_labels (Tensor): Node class labels.
        
        Returns:
            attention_scores (Tensor): The attention scores between nodes.
        """
        # Mask for label similarity
        label_mask = (node_labels.unsqueeze(1) == node_labels.unsqueeze(0)).float()
        
        # Cosine similarity between prototypes
        similarity = F.cosine_similarity(prototypes.unsqueeze(1), prototypes.unsqueeze(0), dim=2)
        
        # Apply softmax on the masked similarity scores to get attention scores
        return F.softmax(similarity * label_mask, dim=-1)

    def select_edges(self, edge_index, edge_probs, attention_scores, node_labels):
        """
        Select edges based on computed probabilities, attention scores, and node labels.
        
        Args:
            edge_index (Tensor): Indices of the edges.
            edge_probs (Tensor): Edge probabilities.
            attention_scores (Tensor): Attention scores based on node labels.
            node_labels (Tensor): Node class labels.
        
        Returns:
            adj_matrix (Tensor): The adjacency matrix representing the graph.
            selected_edges (List): The edges selected for the condensed graph.
        """
        num_nodes = attention_scores.shape[0]  # Number of nodes from attention scores
        device = attention_scores.device
        
        # Initialize adjacency matrix
        adj_matrix = torch.zeros((num_nodes, num_nodes), device=device)
        selected_edges = []
        
        # Generate candidate edges with scores based on probability, attention, and label match
        candidate_edges = []
        for idx in range(edge_index.size(1)):
            src, dst = edge_index[:, idx]
            if src >= num_nodes or dst >= num_nodes:
                continue
                
            # Calculate combined score for each edge
            prob_score = edge_probs[idx]
            attn_score = attention_scores[src, dst]
            label_match = float(node_labels[src] == node_labels[dst])
            
            # Combine the scores for final edge selection
            combined_score = prob_score * attn_score * (1 + 0.5 * label_match)
            candidate_edges.append((src.item(), dst.item(), combined_score.item()))
        
        # Sort edges by score and select top-K edges
        candidate_edges.sort(key=lambda x: x[2], reverse=True)
        k = int(self.rho * len(candidate_edges))
        
        for i, j, score in candidate_edges[:k]:
            if adj_matrix[i, j] == 0 and score > 0:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1  # Ensure undirected edges
                selected_edges.append((i, j))
                
                # Check degree constraint (max degree of nodes in the condensed graph)
                if torch.sum(adj_matrix[i]) > MAX_NODE_DEGREE * self.rho * num_nodes:
                    break
                    
        print(f"Selected {len(selected_edges)} edges with min score {candidate_edges[min(k, len(candidate_edges)-1)][2]:.4f}")
        return adj_matrix, selected_edges

def prepare_condensed_graph(prototypes, embeddings, rho=RHO):
    """
    Prepares the condensed graph by processing node prototypes and edge embeddings.
    
    Args:
        prototypes (dict): A dictionary mapping labels to prototype embeddings.
        embeddings (dict): A dictionary containing edge features and indices.
        rho (float): A parameter controlling edge selection density.
    
    Returns:
        adj_matrix (Tensor): The adjacency matrix of the generated condensed graph.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Process node features and labels
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
    
    # Process edge features and indices
    edge_features = embeddings['edge_features'].to(device)
    edge_index = embeddings['edge_index'].to(device)
    
    # Ensure edge features and indices have the same number of edges
    num_edges = min(edge_index.size(1), edge_features.size(0))
    edge_index = edge_index[:, :num_edges]
    edge_features = edge_features[:num_edges]
    
    # Adjust the edge index to avoid out-of-bound errors
    max_node = node_features.size(0) - 1
    edge_index = torch.clamp(edge_index, 0, max_node)
    
    # Initialize the graph decoder
    decoder = CondensedGraphDecoder(
        prototype_dim=node_features.size(1),
        node_dim=node_features.size(1),  # Keeping dimensionality the same for simplicity
        edge_dim=edge_features.size(1),
        num_classes=len(unique_labels),
        rho=rho
    ).to(device)
    
    # Generate the condensed graph
    adj_matrix, condensed_features, edges = decoder(
        node_features, edge_features, edge_index, node_labels
    )
    
    # Save the results as a condensed graph
    save_condensed_graph(condensed_features, adj_matrix, edges, node_labels)
    return adj_matrix

def save_condensed_graph(node_features, adj_matrix, edges, node_labels, file_name='condensed_graph.pt'):
    """
    Save the condensed graph to a file.
    
    Args:
        node_features (Tensor): The node features of the condensed graph.
        adj_matrix (Tensor): The adjacency matrix of the condensed graph.
        edges (List): The edges selected for the condensed graph.
        node_labels (Tensor): The labels of the nodes in the condensed graph.
        file_name (str): The file name to save the graph.
    """
    edge_index = torch.nonzero(adj_matrix, as_tuple=False).t()
    graph_data = Data(
        x=node_features,
        edge_index=edge_index,
        y=node_labels,
        num_nodes=node_features.size(0)
    )
    torch.save(graph_data, file_name)
    print(f"Condensed graph saved successfully: {node_features.size(0)} nodes, {edge_index.size(1)} edges")

if __name__ == "__main__":
    # Load diffused latent data (prototypes and embeddings)
    prototypes = torch.load('prototypes_diffused.pt')
    embeddings = torch.load('embeddings_diffused.pt')
    
    # Generate the condensed graph
    adj = prepare_condensed_graph(
        prototypes=prototypes,
        embeddings=embeddings,
        rho=RHO 
    )
    
    print("Adjacency matrix statistics:")
    print(f"Total edges: {torch.sum(adj) / 2:.0f}")
