import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class Cheby(nn.Module):
    """Chebyshev Spectral Graph Convolution"""
    def __init__(self, in_channels, out_channels, K, normalization='sym', bias=True):
        super().__init__()
        self.K = K
        self.normalization = normalization
        self.weight = nn.Parameter(torch.Tensor(K+1, in_channels, out_channels))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Compute normalized Laplacian
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        L = -deg_inv_sqrt[row] * deg_inv_sqrt[col]
        L[col == row] += 1
        
        # Perform Chebyshev approximation
        Tx_0 = x
        out = torch.matmul(Tx_0, self.weight[0])
        
        if self.K > 0:
            Tx_1 = torch.sparse.mm(L, x)
            out += torch.matmul(Tx_1, self.weight[1])
        
        for k in range(2, self.K+1):
            Tx_2 = 2 * torch.sparse.mm(L, Tx_1) - Tx_0
            out += torch.matmul(Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2
            
        if self.bias is not None:
            out += self.bias
            
        return out

class GCN(nn.Module):
    """Graph Convolution Network"""
    def __init__(self, in_channels, out_channels, improved=False, cached=False, 
                 add_self_loops=True, normalize=True, bias=True):
        super().__init__()
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col] if self.normalize else 1.0
        
        x = torch.matmul(x, self.weight)
        out = torch.zeros_like(x)
        out = out.index_add_(0, col, x[row] * norm.view(-1, 1))
        
        if self.bias is not None:
            out += self.bias
            
        return out

class SAGE(nn.Module):
    """GraphSAGE"""
    def __init__(self, in_channels, out_channels, aggr='mean', 
                 normalize=False, project=False, bias=True):
        super().__init__()
        self.aggr = aggr
        self.normalize = normalize
        self.project = project
        
        if self.project:
            self.lin = nn.Linear(in_channels, in_channels, bias=False)
            
        self.lin_out = nn.Linear(2*in_channels if aggr == 'lstm' else in_channels*2, 
                                out_channels, bias=bias)
        
        self.reset_parameters()

    def reset_parameters(self):
        if self.project:
            nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.lin_out.weight)
        if self.lin_out.bias is not None:
            nn.init.zeros_(self.lin_out.bias)

    def forward(self, x, edge_index):
        row, col = edge_index
        x_j = x[row]
        
        if self.aggr == 'mean':
            out = torch.zeros(x.size(0), x.size(1), device=x.device)
            out = out.index_add_(0, col, x_j)
            count = degree(col, x.size(0), dtype=x.dtype).view(-1, 1)
            out = out / count.clamp(min=1)
            
        elif self.aggr == 'max':
            out = torch_scatter.scatter_max(x_j, col, dim=0, out=x.new_zeros(x.size()))[0]
            
        elif self.aggr == 'lstm':
            pass  
        
        if self.project:
            out = self.lin(out)
            
        out = torch.cat([x, out], dim=-1)
        out = self.lin_out(out)
        
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
            
        return out

class APPNP(nn.Module):
    """Approximate Personalized Propagation of Neural Predictions"""
    def __init__(self, in_channels, out_channels, hidden_channels, K, alpha, dropout=0.5):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.K = K
        self.alpha = alpha
        self.dropout = dropout

    def forward(self, x, edge_index):
        # Feature transformation
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        
        # Propagation
        h = x
        for _ in range(self.K):
            x = self.propagate(x, edge_index)
            x = x * (1 - self.alpha) + h * self.alpha
        return x

    def propagate(self, x, edge_index):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[col]
        
        out = torch.zeros_like(x)
        out = out.index_add_(0, col, x[row] * norm.view(-1, 1))
        return out

class SGC(nn.Module):
    """Simplifying Graph Convolutions"""
    def __init__(self, in_channels, out_channels, K, bias=True):
        super().__init__()
        self.K = K
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)
        
    def forward(self, x, edge_index):
        h = x
        for _ in range(self.K):
            x = self.propagate(x, edge_index)
        return self.lin(x)
    
    def propagate(self, x, edge_index):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        out = torch.zeros_like(x)
        out = out.index_add_(0, col, x[row] * norm.view(-1, 1))
        return out

class GAT(nn.Module):
    """Graph Attention Network"""
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0.0, add_self_loops=True, bias=True):
        super().__init__()
        self.heads = heads
        self.concat = concat
        self.add_self_loops = add_self_loops
        
        self.weight = nn.Parameter(torch.Tensor(in_channels, heads * out_channels))
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            
        x = torch.matmul(x, self.weight).view(-1, self.heads, self.weight.size(-1)//self.heads)
        
        row, col = edge_index
        x_i = x[row]
        x_j = x[col]
        
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = F.softmax(alpha, dim=1)
        
        if self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
            
        out = torch.zeros(x.size(0), self.heads, x.size(-1), device=x.device)
        out = out.index_add_(0, col, x_i * alpha.unsqueeze(-1))
        
        if self.concat:
            out = out.view(-1, self.heads * x.size(-1))
        else:
            out = out.mean(dim=1)
            
        if self.bias is not None:
            out += self.bias
            
        return out

