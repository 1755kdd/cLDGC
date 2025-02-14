import torch
from torch_geometric.data import Data

def load_dataset(name, root='data'):

    name = name.lower()
    if name == 'cora':
        return _load_cora(root)
    elif name == 'citeseer':
        return _load_citeseer(root)
    elif name == 'ogbn-arxiv':
        return _load_ogbn_arxiv(root)
    elif name == 'reddit':
        return _load_reddit(root)
    elif name == 'flickr':
        return _load_flickr(root)
    else:
        raise ValueError(f"Dataset {name} is not supported. Available datasets: "
                         f"cora, citeseer, ogbn-arxiv, reddit, flickr")

def _load_cora(root):
    from torch_geometric.datasets import Planetoid
    dataset = Planetoid(root=root, name='Cora')
    return dataset[0]

def _load_citeseer(root):
    from torch_geometric.datasets import Planetoid
    dataset = Planetoid(root=root, name='CiteSeer')
    return dataset[0]

def _load_ogbn_arxiv(root):
    from ogb.nodeproppred import NodePropPredDataset

    dataset = NodePropPredDataset(name='ogbn-arxiv', root=root)
    graph, label = dataset[0]

    edge_index = torch.tensor(graph['edge_index'], dtype=torch.long)
    x = torch.tensor(graph['node_feat'], dtype=torch.float)
    y = torch.tensor(label, dtype=torch.long).flatten()

    split_idx = dataset.get_idx_split()
    train_mask = _idx_to_mask(split_idx['train'], y.size(0))
    val_mask = _idx_to_mask(split_idx['valid'], y.size(0))
    test_mask = _idx_to_mask(split_idx['test'], y.size(0))

    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    return data

def _load_reddit(root):
    from torch_geometric.datasets import Reddit
    dataset = Reddit(root=root)
    data = dataset[0]
    return data

def _load_flickr(root):
    try:
        from dgl.data import FlickrDataset
    except ImportError as e:
        raise ImportError(
            "The 'dgl' package is required to load the Flickr dataset. "
            "Please install it via 'conda install -c dglteam dgl' or 'pip install dgl'"
        ) from e

    dataset = FlickrDataset(root=root)
    graph = dataset[0]

    src, dst = graph.edges()
    edge_index = torch.stack([src, dst], dim=0)

    data = Data(
        x=graph.ndata['feat'],
        y=graph.ndata['label'],
        edge_index=edge_index,
        train_mask=graph.ndata['train_mask'],
        val_mask=graph.ndata['val_mask'],
        test_mask=graph.ndata['test_mask']
    )
    return data

def _idx_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask


