import torch
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from Configs.prototype import get_config 
from diffusion.prototypes import *

class PrototypeEngine:
    def __init__(self, dataset: str):
        self.config = get_config(dataset)
        self.device = self.config['device']
        self.rho = self.config['rho']
        self.output_file = self.config['output_file']
        self.batch_size = self.config['batch_size']
        self.clustering_config = self.config.get('clustering', {})
    
    def calculate_structural_signatures(self, edge_index, edge_feats, num_nodes):
        edge_index_np = edge_index.cpu().numpy()
        edge_feats_np = edge_feats.cpu().numpy()

        df = pd.DataFrame(edge_feats_np)
        df['source'] = edge_index_np[0]
        grouped = df.groupby('source').sum()

        signatures = np.zeros((num_nodes, edge_feats.shape[1]))
        valid_indices = grouped.index.astype(int)

        # Ensure valid_indices are within the range of num_nodes
        valid_indices = valid_indices[valid_indices < num_nodes]
        signatures[valid_indices] = grouped.loc[valid_indices].values 

        return torch.FloatTensor(signatures).to(self.device)

    
    def extract_prototypes(self, emb_file: str):
        data = torch.load(emb_file, map_location=self.device)
        
        required_keys = ['node_features', 'edge_features', 'edge_index', 'labels']
        for key in required_keys:
            if key not in data:
                raise KeyError(f"Embedding file missing required key: {key}")
                
        F = data['node_features'].to(self.device)
        edge_feats = data['edge_features'].to(self.device)
        edge_index = data['edge_index'].to(self.device)
        labels = data['labels'].to(self.device)
        
        num_nodes =  F.size(0)
        
        if F.size(0) < num_nodes:
            raise ValueError(f"Node features dimension {F.size(0)} does not match the number of nodes {num_nodes} derived from edge indices.")
        
        s = self.calculate_structural_signatures(edge_index, edge_feats, num_nodes)
        # F_tilde = torch.cat([F, s], dim=1)
        F_tilde = torch.cat([F], dim=1)
        
        prototypes = {}
        unique_labels = torch.unique(labels)
        for class_id in unique_labels:
            mask = labels == class_id
            class_feats = F_tilde[mask]
            Nc = class_feats.size(0)
            Kc = max(1, int(self.rho * Nc))
            
            if Kc == 1:
                prototypes[str(class_id.item())] = class_feats.mean(dim=0, keepdim=True)
            else:
                kmeans = MiniBatchKMeans(
                    n_clusters=Kc,
                    init=self.clustering_config.get('init', 'k-means++'),
                    random_state=self.clustering_config.get('random_state', 42),
                    batch_size=self.batch_size,
                    max_iter=self.clustering_config.get('max_iter', 300),
                    tol=self.clustering_config.get('tolerance', 1e-4)
                )
                kmeans.fit(class_feats.cpu().numpy())
                centers = torch.from_numpy(kmeans.cluster_centers_).float().to(self.device)
                prototypes[str(class_id.item())] = centers
        
        torch.save(prototypes, self.output_file)
        print(f"Successfully saved prototypes to {self.output_file}")
        return prototypes

if __name__ == "__main__":
    dataset_name = "cora"
    engine = PrototypeEngine(dataset_name)
    prototypes = engine.extract_prototypes("Cora_embeddings.pt")
    print(f"Generated {len(prototypes)} class prototypes")
    for cls_id, centers in prototypes.items():
        print(f"Class {cls_id}: {centers.shape[0]} centers")

