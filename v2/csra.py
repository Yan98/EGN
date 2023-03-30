'''
code is based on https://github.com/Kevinz-code/CSRA

'''

import torch_geometric
import torch
import torch_geometric.nn as pyg

class CSRA(torch_geometric.nn.conv.MessagePassing): 
    def __init__(self, dim, T = 0.1):
        super(CSRA, self).__init__()
        self.T = T      # temperature
        self.lam = torch.nn.Parameter(torch.zeros(1) + 0.1)
        self.head = pyg.Linear(dim, dim)
        self.aggr = 'sum'
        
    def forward(self, x_dict,edge_index_dict):
        src = x_dict['example']
        dist = self.head(x_dict['window'])
        edge_index = edge_index_dict[('example','refer','window')]
        alpha = self.edge_updater(edge_index, alpha=(src, dist), edge_attr=None)
        out = self.propagate(edge_index, x=(src,dist), alpha=alpha)
        return dist + self.lam * out
        
    def edge_update(self, alpha_j, alpha_i):
        alpha = (alpha_j - alpha_i) * self.T
        return alpha
    
    def message(self, x_j, alpha,index, size_i=None,ptr=None):
        alpha = torch_geometric.utils.softmax(alpha, index, ptr, size_i)
        return x_j * alpha
