import torch

class Config:

    datasets = ['cora', 'citeseer', 'ogbn-arxiv', 'reddit', 'flickr']
    data_root = './data'
    
    # Dataset-specific configurations
    dataset_config = {
        'cora': {
            'hidden_dim': 1433,
            'output_dim': 1433,
            'edge_proj_hidden': 1433,
            'epochs': 800,
            'lr': 0.001,
            'weight_decay': 5e-4,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        },
        'citeseer': {
            'hidden_dim': 3703,
            'output_dim': 3703,
            'edge_proj_hidden': 3703,
            'epochs': 800,
            'lr': 0.001,
            'weight_decay': 5e-4,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        },
        'ogbn-arxiv': {
            'hidden_dim': 128,
            'output_dim': 128,
            'edge_proj_hidden': 128,
            'epochs': 1000,
            'lr': 0.005,
            'weight_decay': 1e-5,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        },
        'reddit': {
            'hidden_dim': 602,
            'output_dim': 602,
            'edge_proj_hidden': 602,
            'epochs': 600,
            'lr': 0.001,
            'weight_decay': 5e-4,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        },
        'flickr': {
            'hidden_dim': 500,
            'output_dim': 500,
            'edge_proj_hidden': 500,
            'epochs': 500,
            'lr': 0.001,
            'weight_decay': 5e-4,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
    }

    @staticmethod
    def get_config(dataset_name):
        # Returns the configuration for the specified dataset
        if dataset_name in Config.dataset_config:
            return Config.dataset_config[dataset_name]
        else:
            raise ValueError(f"Dataset '{dataset_name}' not found in Config.")
