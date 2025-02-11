# cLDGC

## Data
For cora and citeseer, the code will directly download them using PyG (PyTorch Geometric). For reddit, flickr and ogbn-arxiv, we use the datasets provided by GraphSAINT. They are available on Google Drive link (reddit, flickr and ogbn-arxiv). Download and move them to ./data at your root directory.


## Reproducibility
our codes are implemented step by step:
- run graph_encoder.py for graph encoding pretraining
- run prototype_analysis.py for class-aware prototype extraction
- run diffusion_module.py for conditional diffusion process
- run graph_generation.py for condensed graph generation

- test_condensed_graph.py is the codes for testing
