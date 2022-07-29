import torch
import torch.nn as nn

import dgl
from layers.pe_layer import PELayer

"""
    Graph Transformer with edge features
    
"""
from layers.graph_transformer_edge_layer import GraphTransformerLayer
from layers.mlp_readout_layer import MLPReadout

class GraphTransformerNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = 2
        hidden_dim = net_params['hidden_dim']
        num_heads = net_params['n_heads']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        # self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        self.wl_pos_enc = net_params['wl_pos_enc']
        self.pe_layer = PELayer(net_params)

        # if self.edge_feat:
        #     self.embedding_e = nn.Embedding(num_bond_type, hidden_dim)
        # else:
        self.embedding_e = nn.Linear(1, hidden_dim)
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([ GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                                    self.layer_norm, self.batch_norm, self.residual) for _ in range(n_layers-1) ]) 
        self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm, self.residual))
        # self.MLP_layer = MLPReadout(out_dim, 1)   # 1 out dim since regression problem        

        self.S2S = dgl.nn.pytorch.glob.Set2Set(hidden_dim, 1, 1)
        self.MLP_layer_graph = MLPReadout(hidden_dim*2, 3) # for 3 graph predictions
        self.MLP_layer_nodes = MLPReadout(hidden_dim, 3) # for 3 node predictions
        self.g = None

    def forward(self, g, h, e, pos_enc=None, h_wl_pos_enc=None):
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        h = self.pe_layer(g, h, pos_enc)
        # if not self.edge_feat: # edge feature set to 1
        e = torch.ones(e.size(0),1).to(self.device)
        e = self.embedding_e(e)   
        
        # convnets
        for conv in self.layers:
            h, e = conv(g, h, e)
        g.ndata['h'] = h

        hg = self.S2S(g, h)

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes

        self.g = g
        return self.MLP_layer_nodes(h), self.MLP_layer_graph(hg)

    def loss(self, pred, label):
        nodes_loss = self.single_loss(pred[0], label[0], node_level=True)
        graph_loss = self.single_loss(pred[1], label[1])
        specific_loss = torch.cat((nodes_loss, graph_loss))
        return torch.mean(specific_loss), specific_loss
        
    def single_loss(self, pred, label, node_level=False):
        # for node-level
        if node_level:
            average_nodes = label.shape[0] / self.batch_size
            nodes_loss = (pred - label) ** 2

            # Implementing global add pool of the node losses, reference here
            # https://github.com/cvignac/SMP/blob/62161485150f4544ba1255c4fcd39398fe2ca18d/multi_task_utils/util.py#L99
            self.g.ndata['nodes_loss'] = nodes_loss
            global_add_pool_error = dgl.sum_nodes(self.g, 'nodes_loss') / average_nodes
            loss = torch.mean(global_add_pool_error, dim=0)
            return loss
        
        # for graph-level
        loss = torch.mean((pred - label) ** 2, dim=0)
        return loss
