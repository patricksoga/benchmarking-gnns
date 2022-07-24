import torch
import torch.nn as nn
import scipy as sp
import numpy as np
import networkx as nx
import dgl

from layers.graph_transformer_edge_layer import MultiHeadAttentionLayer
from utils.logging import get_logger

def type_of_enc(net_params):
    learned_pos_enc = net_params.get('learned_pos_enc', False)
    pos_enc = net_params.get('pos_enc', False)
    rand_pos_enc = net_params.get('rand_pos_enc', False)
    if learned_pos_enc:
        return 'learned_pos_enc'
    elif pos_enc:
        return 'pos_enc'
    elif rand_pos_enc:
        return 'rand_pos_enc'
    else:
        return 'none'

class PELayer(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.device = net_params['device']
        self.pos_enc = net_params.get('pos_enc', False)
        self.learned_pos_enc = net_params.get('learned_pos_enc', False)
        self.rand_pos_enc = net_params.get('rand_pos_enc', False)
        self.pos_enc_dim = net_params.get('pos_enc_dim', 0)
        self.wl_pos_enc = net_params.get('wl_pos_enc', False)
        self.dataset = net_params.get('dataset', 'CYCLES')
        self.pow_of_mat = net_params.get('pow_of_mat', 1)

        self.matrix_type = net_params['matrix_type']
        self.logger = get_logger(net_params['log_file'])
        hidden_dim = net_params['hidden_dim']
        max_wl_role_index = 37 # this is maximum graph size in the dataset

        self.logger.info(type_of_enc(net_params))
        if self.pos_enc:
            # logger.info("Using Laplacian position encoding")
            self.embedding_pos_enc = nn.Linear(self.pos_enc_dim, hidden_dim)
        elif self.learned_pos_enc or self.rand_pos_enc:
            # logger.info("Using automata position encoding")
            self.pos_initial = nn.Parameter(torch.Tensor(self.pos_enc_dim, 1), requires_grad=not self.rand_pos_enc)
            self.pos_transition = nn.Parameter(torch.Tensor(self.pos_enc_dim, self.pos_enc_dim), requires_grad=not self.rand_pos_enc)
            nn.init.normal_(self.pos_initial)
            nn.init.orthogonal_(self.pos_transition)
            self.embedding_pos_enc = nn.Linear(self.pos_enc_dim, hidden_dim)

            # self.mat_pows = nn.Parameter(torch.Tensor(size=(1,)))
            # nn.init.constant_(self.mat_pows, 1)
            self.mat_pows = nn.ParameterList([nn.Parameter(torch.Tensor(size=(1,))) for _ in range(self.pow_of_mat)])
            for mat_pow in self.mat_pows:
                nn.init.constant_(mat_pow, 1)

        in_dim = 1
        if self.dataset == "SBM_PATTERN":
            in_dim = net_params['in_dim']
        self.embedding_h = nn.Linear(in_dim, hidden_dim)

        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)

        self.use_pos_enc = self.pos_enc or self.wl_pos_enc or self.learned_pos_enc or self.rand_pos_enc
        if self.use_pos_enc:
            self.logger.info(f"Using {self.pos_enc_dim} dimension positional encoding (# states if an automata enc, otherwise smallest k eigvecs)")
        
        self.logger.info(f"Using matrix: {self.matrix_type}")
        self.logger.info(f"Matrix power: {self.pow_of_mat}")


    def forward(self, g, h, pos_enc=None, h_wl_pos_enc=None):
        if self.wl_pos_enc:
            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc) 
            h = h + h_wl_pos_enc
            return h

        pe = None
        if self.pos_enc:
            pe = self.embedding_pos_enc(pos_enc)
        elif self.learned_pos_enc:
            # mat = g.adjacency_matrix().to_dense().to(self.device)
            # mat = self.type_of_matrix(g, self.matrix_type, self.pow_of_mat)
            mat = self.type_of_matrix(g, self.matrix_type, self.mat_pows)
            z = torch.zeros(self.pos_enc_dim, g.num_nodes()-1, requires_grad=False).to(self.device)
            vec_init = torch.cat((self.pos_initial, z), dim=1).to(self.device)
            vec_init = vec_init.transpose(1, 0).flatten()
            kron_prod = torch.kron(mat.reshape(mat.shape[1], mat.shape[0]), self.pos_transition).to(self.device)
            # dim0, dim1 = mat.shape[0]*self.pos_enc_dim, mat.shape[1]*self.pos_enc_dim
            # kron_prod = torch.einsum('ik,jl', mat.t(), self.pos_transition).reshape(dim0, dim1)
            B = torch.eye(kron_prod.shape[1]).to(self.device) - kron_prod

            encs = torch.linalg.solve(B, vec_init)
            stacked_encs = torch.stack(encs.split(self.pos_enc_dim), dim=1)
            stacked_encs = stacked_encs.transpose(1, 0)
            pe = self.embedding_pos_enc(stacked_encs)
        elif self.rand_pos_enc:
            z = torch.zeros(self.pos_enc_dim, g.num_nodes()-1, requires_grad=False).to(torch.device('cpu'))
            vec_init = torch.cat((self.pos_initial.to(torch.device('cpu')), z), dim=1).to(torch.device('cpu'))
            # mat = g.adjacency_matrix().to_dense().to(torch.device('cpu'))
            mat = self.type_of_matrix(g, self.matrix_type, self.pow_of_mat)
            transition_inv = torch.inverse(self.pos_transition).to(torch.device('cpu'))

            # AX + XB = Q
            #  X = alpha
            #  A = mu inverse
            #  B = -A
            #  Q = mu inverse + pi
            transition_inv = transition_inv.numpy()
            # mat = mat.numpy()
            mat = mat.cpu().numpy()
            vec_init = vec_init.numpy()
            pe = sp.linalg.solve_sylvester(transition_inv, -mat, transition_inv @ vec_init)
            pe = torch.from_numpy(pe.T).to(self.device)
            pe = self.embedding_pos_enc(pe)
        else:
            if self.dataset == "ZINC":
                pe = h
            # elif self.dataset == "CYCLES":
            #     pe = self.embedding_h(h)
        
        if self.dataset in ("CYCLES", "ZINC"):
            return pe
        # return h + pe if pe is not None else h
        return pe

    def get_normalized_laplacian(self, g):
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        N = sp.sparse.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        L = sp.sparse.eye(g.number_of_nodes()) - N * A * N
        return L

    def type_of_matrix(self, g, matrix_type, pow):
        """
        Takes a DGL graph and returns the type of matrix to use for the layer.
            'A': adjacency matrix (default),
            'L': Laplacian matrix,
            'NL': normalized Laplacian matrix,
            'E': eigenvector matrix,
        """
        matrix = g.adjacency_matrix().to_dense().to(self.device)
        if matrix_type == 'A':
            matrix = g.adjacency_matrix().to_dense().to(self.device)
        elif matrix_type == 'NL':
            laplacian = self.get_normalized_laplacian(g)
            matrix = torch.from_numpy(laplacian.A).float().to(self.device) 
        elif matrix_type == "L":
            graph = g.cpu().to_networkx().to_undirected()
            matrix = torch.from_numpy(nx.laplacian_matrix(graph).A).to(self.device)
        elif matrix_type == "E":
            laplacian = self.get_normalized_laplacian(g)
            EigVal, EigVec = np.linalg.eig(laplacian.toarray())
            idx = EigVal.argsort() # increasing order
            EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
            matrix = torch.from_numpy(EigVec).float().to(self.device)

        
        # learnable adj matrix "masks"
        # matrix = torch.matrix_power(matrix, pow)
        matrices = [torch.matrix_power(matrix, i) for i in range(self.pow_of_mat)]

        # multiply each matrix by the corresponding mat_pow
        for i in range(len(matrices)):
            matrices[i] = matrices[i] * self.mat_pows[i]
        matrix = torch.sum(torch.stack(matrices, dim=0), dim=0)

        return matrix

