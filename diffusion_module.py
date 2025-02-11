import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalDiffusion(nn.Module):
    def __init__(self, feat_dim, label_dim, hidden_dim=256):
        super().__init__()
        self.timesteps = 1000
        
        self.register_buffer('betas', torch.linspace(1e-4, 0.02, self.timesteps))
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alpha_bars', torch.cumprod(self.alphas, dim=0))
        
        self.label_emb = nn.Embedding(label_dim, hidden_dim)
        
        self.node_net = nn.Sequential(
            nn.Linear(feat_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim)
        )
        self.edge_net = nn.Sequential(
            nn.Linear(feat_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim)
        )

    def forward(self, x, h, y, t):
        y_emb = self.label_emb(y)
        x_in = torch.cat([x, y_emb], dim=-1)
        h_in = torch.cat([h, y_emb], dim=-1)
        return self.node_net(x_in), self.edge_net(h_in)

    def diffuse(self, x, h, y):
        device = x.device
        batch_size = x.size(0)
        
        if h.size(0) != batch_size:
            h = h[torch.arange(batch_size) % h.size(0)]
        
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        alpha_bars = self.alpha_bars[t][:, None]
        
        sqrt_alpha = torch.sqrt(alpha_bars)
        sqrt_one_minus = torch.sqrt(1. - alpha_bars)
        
        eps_x = torch.randn_like(x)
        eps_h = torch.randn_like(h)
        
        x_noisy = sqrt_alpha * x + sqrt_one_minus * eps_x
        h_noisy = sqrt_alpha * h + sqrt_one_minus * eps_h
        
        return x_noisy, h_noisy, eps_x, eps_h, t

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    embeddings = torch.load('embeddings.pt', map_location=device)
    prototypes = torch.load('prototypes.pt', map_location=device)
    
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
    
    def align_edges(h, target_size):
        repeat_times = (target_size // h.size(0)) + 1
        return h.repeat(repeat_times, 1)[:target_size]
    
    H = align_edges(H_original, P.size(0))  # [70, 1433]
    
    model = ConditionalDiffusion(feat_dim=1433, label_dim=7).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 1000
    batch_size = 32
    
    for epoch in range(epochs):
        idx = torch.randperm(P.size(0), device=device)[:batch_size]
        x_batch = P[idx]
        y_batch = Y[idx]
        
        h_batch = align_edges(H_original, batch_size)
        
        x_noisy, h_noisy, eps_x, eps_h, t = model.diffuse(x_batch, h_batch, y_batch)
        pred_eps_x, pred_eps_h = model(x_noisy, h_noisy, y_batch, t)
        
        loss = F.mse_loss(pred_eps_x, eps_x) + F.mse_loss(pred_eps_h, eps_h)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
    
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
    
    torch.save(prototypes_diffused, 'prototypes_diffused.pt')
    torch.save(embeddings_diffused, 'embeddings_diffused.pt')

if __name__ == '__main__':
    main()