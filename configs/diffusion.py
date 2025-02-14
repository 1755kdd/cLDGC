
datasets_config = {
    'cora': {
        'feat_dim': 1433,
        'label_dim': 7,
        'embeddings_path': 'Cora_embeddings.pt',
        'prototypes_path': 'prototypes_cora.pt',
        'epochs': 1000,
        'batch_size': 32,
        'lr': 1e-3
    },
    'citeseer': {
        'feat_dim': 3703,
        'label_dim': 6,
        'embeddings_path': 'Citeseer_embeddings.pt',
        'prototypes_path': 'prototypes_citeseer.pt',
        'epochs': 1000,
        'batch_size': 32,
        'lr': 1e-3
    },
    'ogbn-arxiv': {
        'feat_dim': 128,
        'label_dim': 40,
        'embeddings_path': 'ogbn-arxiv_embeddings.pt',
        'prototypes_path': 'prototypes_ogbn-arxiv.pt',
        'epochs': 1000,
        'batch_size': 32,
        'lr': 1e-3
    },
    'reddit': {
        'feat_dim': 602,
        'label_dim': 41,
        'embeddings_path': 'reddit_embeddings.pt',
        'prototypes_path': 'prototypes_reddit.pt',
        'epochs': 1000,
        'batch_size': 32,
        'lr': 1e-3
    },
    'flickr': {
        'feat_dim': 500,
        'label_dim': 7,
        'embeddings_path': 'flickr_embeddings.pt',
        'prototypes_path': 'prototypes_flickr.pt',
        'epochs': 1000,
        'batch_size': 32,
        'lr': 1e-3
    }
}



