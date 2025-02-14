import torch
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


class StructuralSignatures:
    def __init__(self, device):
        self.device = device

    def calculate_structural_signatures(self, edge_index, edge_feats, num_nodes):
        """
        Calculates the structural signatures based on edge features and edge indices.
        :param edge_index: Torch tensor containing the edge indices.
        :param edge_feats: Torch tensor containing the edge features.
        :param num_nodes: The number of nodes in the graph.
        :return: The structural signatures as a torch tensor.
        """
        edge_index_np = edge_index.cpu().numpy()
        edge_feats_np = edge_feats.cpu().numpy()

        # Create a DataFrame to sum the edge features by source node
        df = pd.DataFrame(edge_feats_np)
        df['source'] = edge_index_np[0]
        grouped = df.groupby('source').sum()

        # Initialize the signature matrix
        signatures = np.zeros((num_nodes, edge_feats.shape[1]))
        valid_indices = grouped.index.astype(int)

        # Ensure valid indices are within the range of num_nodes
        valid_indices = valid_indices[valid_indices < num_nodes]
        signatures[valid_indices] = grouped.loc[valid_indices].values

        return torch.FloatTensor(signatures).to(self.device)


class PrototypeExtractor:
    def __init__(self, rho, clustering_config, batch_size, device, output_file):
        self.rho = rho
        self.clustering_config = clustering_config
        self.batch_size = batch_size
        self.device = device
        self.output_file = output_file

    def extract_prototypes(self, F_tilde, labels):
        """
        Extracts prototypes by performing clustering on each class's feature set.
        :param F_tilde: The feature matrix concatenating node features and structural signatures.
        :param labels: The labels for each node.
        :return: A dictionary where each key is a class label, and the value is the prototype(s).
        """
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
                # Perform clustering using MiniBatchKMeans
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

        return prototypes


class PrototypeEngine:
    def __init__(self, dataset, get_config_fn):
        """
        The main engine to extract prototypes from embeddings.
        :param dataset: The name of the dataset.
        :param get_config_fn: A function to get the configuration dictionary.
        """
        self.config = get_config_fn(dataset)
        self.device = self.config['device']
        self.rho = self.config['rho']
        self.output_file = self.config['output_file']
        self.batch_size = self.config['batch_size']
        self.clustering_config = self.config.get('clustering', {})
        
        # Initialize the necessary components
        self.structural_signatures = StructuralSignatures(self.device)
        self.prototype_extractor = PrototypeExtractor(
            self.rho, self.clustering_config, self.batch_size, self.device, self.output_file)

    def extract_prototypes(self, emb_file):
        """
        Main method to extract prototypes from an embedding file.
        :param emb_file: Path to the embedding file.
        :return: A dictionary of class prototypes.
        """
        data = torch.load(emb_file, map_location=self.device)
        
        required_keys = ['node_features', 'edge_features', 'edge_index', 'labels']
        for key in required_keys:
            if key not in data:
                raise KeyError(f"Embedding file missing required key: {key}")
                
        F = data['node_features'].to(self.device)
        edge_feats = data['edge_features'].to(self.device)
        edge_index = data['edge_index'].to(self.device)
        labels = data['labels'].to(self.device)
        
        num_nodes = F.size(0)
        
        if F.size(0) < num_nodes:
            raise ValueError(f"Node features dimension {F.size(0)} does not match the number of nodes {num_nodes} derived from edge indices.")
        
        # Calculate structural signatures
        s = self.structural_signatures.calculate_structural_signatures(edge_index, edge_feats, num_nodes)
        
        # Concatenate node features and structural signatures
        F_tilde = torch.cat([F, s], dim=1)
        
        # Extract prototypes using clustering
        prototypes = self.prototype_extractor.extract_prototypes(F_tilde, labels)
        
        # Save the prototypes
        torch.save(prototypes, self.output_file)
        print(f"Successfully saved prototypes to {self.output_file}")
        return prototypes
