import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from diffusion.conditional_diffusion import ConditionalLatentDiffusion
import argparse
from  Configs.diffusion import datasets_config

def align_edges(h, target_size):
    repeat_times = (target_size // h.size(0)) + 1
    return h.repeat(repeat_times, 1)[:target_size]

def train_diffusion(dataset_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset config
    config = datasets_config[dataset_name]
    
    # Load data
    embeddings = torch.load(config['embeddings_path'], map_location=device)
    prototypes = torch.load(config['prototypes_path'], map_location=device)
    
    # Process prototypes
    P_list, Y_list = [], []
    for key in prototypes:
        if str(key).isdigit():
            tensor = prototypes[key].to(device)
            P_list.append(tensor)
            Y_list.append(torch.full((tensor.size(0),), int(key), device=device))
    
    if not P_list:
        raise ValueError("Invalid prototypes data")
    
    P = torch.cat(P_list, dim=0)        
    Y = torch.cat(Y_list, dim=0)      
    H_original = embeddings['edge_features'].to(device) 
    
    # Initialize model
    model = ConditionalLatentDiffusion(
        feat_dim=config['feat_dim'],
        label_dim=config['label_dim']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    # Training loop
    for epoch in range(config['epochs']):
        idx = torch.randperm(P.size(0), device=device)[:config['batch_size']]
        x_batch = P[idx]
        y_batch = Y[idx]
        
        h_batch = align_edges(H_original, config['batch_size'])
        
        x_noisy, h_noisy, eps_x, eps_h, t = model.diffuse(x_batch, h_batch, y_batch)
        pred_eps_x, pred_eps_h = model(x_noisy, h_noisy, y_batch, t)
        
        loss = F.mse_loss(pred_eps_x, eps_x) + F.mse_loss(pred_eps_h, eps_h)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Dataset: {dataset_name} | Epoch {epoch} | Loss: {loss.item():.4f}")
    
    # Save results
    model.eval()
    with torch.no_grad():
        H_full = align_edges(H_original, P.size(0))
        x_noisy, h_noisy, eps_x, eps_h, t = model.diffuse(P, H_full, Y)
        pred_eps_x, pred_eps_h = model(x_noisy, h_noisy, Y, 
                                     torch.full((P.size(0),), model.timesteps-1, device=device))
        
        P_diffused = x_noisy - pred_eps_x
        H_diffused = h_noisy - pred_eps_h
    
    prototypes_diffused = {}
    start = 0
    for key in prototypes:
        if str(key).isdigit():
            end = start + prototypes[key].size(0)
            prototypes_diffused[key] = P_diffused[start:end].cpu()
            start = end
    
    embeddings_diffused = {
        'node_features': P_diffused.cpu(),
        'edge_features': H_diffused[:H_original.size(0)].cpu(),
        'edge_index': embeddings['edge_index'].cpu(),
        'labels': embeddings['labels'].cpu()
    }
    
    # Save with dataset name
    torch.save(prototypes_diffused, f'prototypes_diffused_{dataset_name}.pt')
    torch.save(embeddings_diffused, f'embeddings_diffused_{dataset_name}.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['cora', 'citeseer', 'ogbn-arxiv', 'reddit', 'flickr'])
    args = parser.parse_args()
    
    train_diffusion(args.dataset)
    