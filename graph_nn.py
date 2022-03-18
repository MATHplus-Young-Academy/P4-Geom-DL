import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
import torch_geometric.nn as gnn
from pprint import pprint


class GNN(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__(**kwargs)
        self.mlp = pyg.nn.MLP([in_dim, 32, 32, out_dim])
        self.conv1 = gnn.PPFConv(self.mlp)
        
    def forward(self, graph):
        x, pos, norm, edge_index = graph.x, graph.pos, graph.norm, graph.edge_index
        x = self.conv1(x, pos, norm, edge_index)
        return F.softmax(x, dim=0)
    

def train_loop(dataloader, model, optim, loss_fn):
    total_loss, num_batches = 0.0, len(dataloader)
    #pprint(num_batches)
    for k, batch in enumerate(dataloader):
        pred = model(batch)
#        pprint(pred)
        loss = loss_fn(pred[:,0], batch.y)
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        total_loss += loss.item()
        
    return total_loss / num_batches