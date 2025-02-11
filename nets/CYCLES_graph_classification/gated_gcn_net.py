import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

from layers.pe_layer import PELayer

"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""
from layers.gated_gcn_layer import GatedGCNLayer
from layers.mlp_readout_layer import MLPReadout

class GatedGCNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        batch_norm = net_params['batch_norm']
        layer_norm = net_params['layer_norm']
        self.readout = net_params['readout']
        self.residual = net_params['residual']
        self.device = net_params['device']
        self.pe_layer = PELayer(net_params)

        self.embedding_e = nn.Linear(1, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([ GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                       batch_norm, layer_norm, self.residual) for _ in range(n_layers-1) ]) 
        self.layers.append(GatedGCNLayer(hidden_dim, out_dim, dropout, batch_norm, self.layer_norm, self.residual))
        self.MLP_layer = MLPReadout(out_dim, n_classes)

        
    def forward(self, g, h, e, pos_enc=None):

        # input embedding
        h = self.pe_layer(g, h, pos_enc)
        h = self.in_feat_dropout(h)
        
        # edge feature set to 1
        e = torch.ones(e.size(0),1).to(self.device) 
        e = self.embedding_e(e)
        
        # convnets
        for conv in self.layers:
            h, e = conv(g, h, e)
        g.ndata['h'] = h
        
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
            
        return self.MLP_layer(hg)
        
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss
