import torch
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

class PrototypeEngine:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
    
    def calculate_structural_signatures(self, edge_index, edge_feats, num_nodes):
        edge_index_np = edge_index.cpu().numpy()
        edge_feats_np = edge_feats.cpu().numpy()
        
        df = pd.DataFrame(edge_feats_np)
        df['source'] = edge_index_np[0]
        grouped = df.groupby('source').sum()
        
        signatures = np.zeros((num_nodes, edge_feats.shape[1]))
        valid_indices = grouped.index.values
        signatures[valid_indices] = grouped.values
        
        return torch.FloatTensor(signatures).to(self.device)
    
    def extract_prototypes(self, emb_file, rho=0.1, output_file='prototypes.pt'):
        data = torch.load(emb_file)
        F = data['node_features'].to(self.device)
        edge_feats = data['edge_features'].to(self.device)
        edge_index = data['edge_index'].to(self.device)
        labels = data['labels'].to(self.device)
        
        num_nodes = F.size(0)
        s = self.calculate_structural_signatures(edge_index, edge_feats, num_nodes)
        
        # F_tilde = torch.cat([F, s], dim=1)
        F_tilde = torch.cat([F], dim=1)
        prototypes = {}

        for class_id in torch.unique(labels):
            mask = labels == class_id
            class_feats = F_tilde[mask]
            Nc = class_feats.size(0)
            
            Kc = max(1, int(torch.floor(torch.tensor(rho * Nc))))
            
            if Kc == 1:
                prototypes[str(class_id.item())] = class_feats.mean(dim=0, keepdim=True)
            else:
                try:
                    from cuml.cluster import KMeans
                    kmeans = KMeans(n_clusters=Kc, init='k-means++', random_state=77)
                    kmeans.fit(class_feats.cpu().numpy())
                    centers = torch.from_numpy(kmeans.cluster_centers_).to(self.device)
                except ImportError:
                    kmeans = MiniBatchKMeans(n_clusters=Kc, init='k-means++', 
                                           random_state=77, batch_size=1024)
                    kmeans.fit(class_feats.cpu().numpy())
                    centers = torch.from_numpy(kmeans.cluster_centers_).float().to(self.device)
                
                prototypes[str(class_id.item())] = centers
        
        torch.save(prototypes, output_file)
        print(f"Prototypes saved to {output_file}")
        return prototypes

if __name__ == "__main__":
    engine = PrototypeEngine()
    engine.extract_prototypes('embeddings.pt', rho=0.5)
