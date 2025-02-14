# cLDGC: Conditional Latent Diffusion-based Graph Condensation

## Overview
cLDGC presents an architecture-agnostic generative framework for synthesizing condensed graph representations through synergistic optimization of feature-structure interdependencies. Our novel approach employs a four-stage process to eliminate architectural biases while preserving task-critical graph semantics:

1. **Latent Space Encoding**: Pre-train a graph encoder to project node attributes and topological relationships into a shared latent representation
2. **Prototype Extraction**: Extract graph prototypes via clustering on learned representations, disentangling essential topological patterns
3. **Conditional Diffusion Process**: Incorporate task-specific semantics through label-conditioned diffusion processes with adaptive noise scheduling
4. **Condensed Graph Generation**: Reconstruct condensed feature matrices and adjacency structures from refined latent embeddings in unified generation step

# Quick Start
## Create environment
```bash
- Python >=3.9
- Pytorch >= 1.12.1
```

## Running cLDGC
### Pretrain graph encoder
```bash
python train_encoder.py
```

### Protorype extraction
```bash
python extract_prototype.py
```

### Diffusion Process
```bash
python train_diffusion.py
```

### Condensed graph generation
```bash
python generate_graph.py
```
