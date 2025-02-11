import torch
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.rcParams.update({
    'font.size': 20,      
    'axes.labelsize': 22,  
    'legend.fontsize': 18,  
    'xtick.labelsize': 18,  
    'ytick.labelsize': 18, 
    'pdf.fonttype': 42,   
    'ps.fonttype': 42   
})


os.makedirs('./figures', exist_ok=True)


prototypes = torch.load('prototypes.pt')
embeddings_data = torch.load('embeddings.pt')


prototypes_list = []
proto_labels = []
for label in prototypes.keys():
    prototypes_list.append(prototypes[label])
    proto_labels.extend([int(label)] * prototypes[label].shape[0])

prototypes_tensor = torch.cat(prototypes_list, dim=0).cpu() 
proto_labels = np.array(proto_labels, dtype=np.int64)


node_features = embeddings_data['node_features'].cpu()
node_labels = embeddings_data['labels'].numpy().astype(np.int64)


all_embeddings = torch.cat([node_features, prototypes_tensor], dim=0)
all_labels = np.concatenate([node_labels, proto_labels])


tsne = TSNE(n_components=2, random_state=42, perplexity=30) 
embeddings_2d = tsne.fit_transform(all_embeddings.numpy())


node_2d = embeddings_2d[:len(node_labels)]
proto_2d = embeddings_2d[len(node_labels):]


plt.figure(figsize=(14, 10))  
colors = plt.cm.get_cmap('tab10', len(np.unique(all_labels)))


node_scatter = plt.scatter(
    node_2d[:, 0], node_2d[:, 1],
    c=node_labels,
    cmap=colors,
    s=100,           
    alpha=0.85,     
    edgecolors='none', 
    linewidths=0,
    zorder=1,
    label='Nodes'
)

proto_markers = []
for label in np.unique(proto_labels):
    mask = (proto_labels == label)
    if np.any(mask):
        proto_scatter = plt.scatter(
            proto_2d[mask, 0], proto_2d[mask, 1],
            c=[colors(label % colors.N)],
            marker='*',        
            s=1200,             
            edgecolors='k',    
            linewidths=0,     
            zorder=3,         
            label=f'Proto {label}'
        )
        proto_markers.append(proto_scatter)

plt.tight_layout()

plt.savefig('./figures/visualization.eps', 
           format='eps',
           bbox_inches='tight',
           dpi=500,
           facecolor='white')

plt.savefig('./figures/visualization.pdf', 
           format='pdf',
           bbox_inches='tight')

plt.close()