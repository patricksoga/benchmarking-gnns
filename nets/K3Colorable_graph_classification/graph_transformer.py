import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    Graph Transformer with edge features
    
"""
from layers.graph_transformer_edge_layer import GraphTransformerLayer
from layers.mlp_readout_layer import MLPReadout

class GraphTransformerNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        hidden_dim = net_params['hidden_dim']
        n_classes = net_params['n_classes']
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
        self.pos_enc = net_params['pos_enc']
        self.wl_pos_enc = net_params['wl_pos_enc']
        self.learned_pos_enc = net_params.get('learned_pos_enc', False)

        max_wl_role_index = 37 # this is maximum graph size in the dataset
        pos_enc_dim = net_params['pos_enc_dim']
        self.pos_enc_dim = pos_enc_dim
        if self.pos_enc:
            # pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        elif self.learned_pos_enc:
            self.pos_initial = nn.Parameter(torch.Tensor(pos_enc_dim, 1))
            self.pos_transition = nn.Parameter(torch.Tensor(pos_enc_dim, pos_enc_dim))
            nn.init.normal_(self.pos_initial)
            nn.init.orthogonal_(self.pos_transition)
            self.embedding_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        elif self.wl_pos_enc:
            self.embedding_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)

        in_dim = 1
        self.embedding_h = nn.Linear(in_dim, hidden_dim)

        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)
        
        # if self.edge_feat:
        #     self.embedding_e = nn.Embedding(num_bond_type, hidden_dim)
        # else:
        self.embedding_e = nn.Linear(1, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([ GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                                    self.layer_norm, self.batch_norm, self.residual) for _ in range(n_layers-1) ]) 
        self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm, self.residual))
        # self.MLP_layer = MLPReadout(out_dim, 1)   # 1 out dim since regression problem        
        self.MLP_layer = MLPReadout(out_dim, n_classes)
        
    def forward(self, g, h, e, pos_enc=None, h_wl_pos_enc=None):

        # input embedding
        # h = self.embedding_h(h)
        # h = self.in_feat_dropout(h)
        if self.pos_enc:
            # pos_enc = self.embedding_pos_enc(pos_enc) 
            # h = h + pos_enc
            h = self.embedding_pos_enc(pos_enc)
        elif self.learned_pos_enc:
            A = g.adjacency_matrix().to_dense().to(self.device)
            z = torch.zeros(self.pos_enc_dim, g.num_nodes()-1, requires_grad=False).to(self.device)
            vec_init = torch.cat((self.pos_initial, z), dim=1).to(self.device)
            vec_init = vec_init.transpose(1, 0).flatten()
            kron_prod = torch.kron(A.t().contiguous(), self.pos_transition).to(self.device)
            B = torch.eye(kron_prod.shape[1]).to(self.device) - kron_prod
            encs = torch.linalg.solve(B, vec_init)
            stacked_encs = torch.stack(encs.split(self.pos_enc_dim), dim=1).transpose(1, 0)
            h = self.embedding_pos_enc(stacked_encs)
        else:
            h = self.embedding_h(h)
        if self.wl_pos_enc:
            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc) 
            h = h + h_wl_pos_enc
        # if not self.edge_feat: # edge feature set to 1
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
