# First version: 19th of March 2022
# Author: Felix Herter, Nikolas Tapia
# Copyright 2022 Weierstrass Institute
# Copyright 2022 Zuse Institute Berlin
# 
#    This software was developed during the Math+ "Maths meets Image" hackathon 2022.
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
# 
#        http://www.apache.org/licenses/LICENSE-2.0
# 
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
import torch_geometric.nn as gnn
from pprint import pprint
from tqdm import tqdm_notebook


class GNN(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__(**kwargs)
        self.mlp1 = pyg.nn.MLP([in_dim, 8, 16])
        self.mlp2 = pyg.nn.MLP([20, 32, 16])
        self.mlp3 = pyg.nn.MLP([20, 32, 16])
        self.mlp = pyg.nn.MLP([16, 32, 16])
        self.out_mlp = pyg.nn.MLP([16, 32, out_dim])
        self.conv1 = gnn.PPFConv(self.mlp1, self.mlp)
        self.conv2 = gnn.PPFConv(self.mlp2, self.mlp)
        self.conv3 = gnn.PPFConv(self.mlp3, self.out_mlp)
        
    def forward(self, graph):
        x, pos, norm, edge_index = graph.x, graph.pos, graph.norm, graph.edge_index
        x = self.conv1(x, pos, norm, edge_index)
        x = F.relu(x)
        x = self.conv2(x, pos, norm, edge_index)
        x = F.relu(x)
        x = self.conv3(x, pos, norm, edge_index)
        x = F.relu(x)
        return F.softmax(x.flatten(), dim=0)

class ShallowGNN(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__(**kwargs)
        self.mlp1 = pyg.nn.MLP([in_dim, 32, 32])
        self.out_mlp = pyg.nn.MLP([32, 32, out_dim])
        self.conv1 = gnn.PPFConv(self.mlp1, self.out_mlp)
        
    def forward(self, graph):
        x, pos, norm, edge_index = graph.x, graph.pos, graph.norm, graph.edge_index
        x = self.conv1(x, pos, norm, edge_index)
        x = F.relu(x)
        return F.softmax(x.flatten(), dim=0)
    

def train_loop(dataloader, model, optim, loss_fn, device):
    total_loss, num_batches = 0.0, len(dataloader)
    for k, batch in enumerate(dataloader):
        batch.to(device=device)
        pred = model(batch)
        loss = loss_fn(pred, batch.y)
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        total_loss += loss.item()
        
    return total_loss / num_batches