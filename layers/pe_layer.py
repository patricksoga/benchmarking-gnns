import torch
import torch.nn as nn
import scipy as sp
import numpy as np
import networkx as nx
import dgl
import scipy

from utils.main_utils import get_logger

def type_of_enc(net_params):
    learned_pos_enc = net_params.get('learned_pos_enc', False)
    pos_enc = net_params.get('pos_enc', False)
    adj_enc = net_params.get('adj_enc', False)
    rand_pos_enc = net_params.get('rand_pos_enc', False)
    partial_rw_pos_enc = net_params.get('partial_rw_pos_enc', False)
    spectral_attn = net_params.get('spectral_attn', False)
    n_gape = net_params.get('n_gape', 1)
    if learned_pos_enc:
        return 'learned_pos_enc'
    elif pos_enc:
        return 'pos_enc'
    elif adj_enc:
        return 'adj_enc'
    elif rand_pos_enc:
        print(f'using {str(n_gape)} automata/automaton')
        return f'rand_pos_enc'
    elif partial_rw_pos_enc:
        return 'partial_rw_pos_enc'
    elif spectral_attn:
        return 'spectral_attn'
    else:
        return 'no_pe'

class PELayer(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.device = net_params['device']
        self.pos_enc = net_params.get('pos_enc', False)
        self.learned_pos_enc = net_params.get('learned_pos_enc', False)
        self.rand_pos_enc = net_params.get('rand_pos_enc', False)
        self.rw_pos_enc = net_params.get('rw_pos_enc', False) or net_params.get('partial_rw_pos_enc', False)
        self.adj_enc = net_params['adj_enc']
        self.pos_enc_dim = net_params.get('pos_enc_dim', 0)
        self.wl_pos_enc = net_params.get('wl_pos_enc', False)
        self.dataset = net_params.get('dataset', 'CYCLES')
        self.pow_of_mat = net_params.get('pow_of_mat', 1)
        self.num_initials = net_params.get('num_initials', 1)
        self.pagerank = net_params.get('pagerank', False)
        self.cat = net_params.get('cat_gape', False)
        self.n_gape = net_params.get('n_gape', 1)
        self.gape_pooling = net_params.get('gape_pooling', 'mean')
        self.gape_softmax_after = net_params.get('gape_softmax_after', False)
        self.gape_softmax_before = net_params.get('gape_softmax_before', False)
        self.gape_individual = net_params.get('gape_individual', False)
        self.clamp = net_params.get('gape_clamp', False)

        self.matrix_type = net_params['matrix_type']
        self.logger = get_logger(net_params['log_file'])

        self.power_method = net_params.get('power_method', False)
        self.power_method_iters = net_params.get('power_iters', 50)

        self.rand_sketchy_pos_enc = net_params.get('rand_sketchy_pos_enc', False)

        hidden_dim = net_params['hidden_dim']
        max_wl_role_index = 37 # this is maximum graph size in the dataset

        self.logger.info(type_of_enc(net_params))
        if self.pos_enc:
            self.embedding_pos_enc = nn.Linear(self.pos_enc_dim, hidden_dim)
        if self.adj_enc:
            self.embedding_pos_enc = nn.Linear(self.pos_enc_dim, hidden_dim)
        elif self.learned_pos_enc or self.rand_pos_enc or self.rand_sketchy_pos_enc:

            # if net_params['diag']:
            #     self.pos_transition = nn.Parameter(torch.Tensor(self.pos_enc_dim), requires_grad=not self.rand_pos_enc)
            #     nn.init.normal_(self.pos_transition)
            # else:
            #     self.pos_transition = nn.Parameter(torch.Tensor(self.pos_enc_dim, self.pos_enc_dim), requires_grad=not self.rand_pos_enc)
            #     nn.init.orthogonal_(self.pos_transition)

            # init initial vectors

            # self.pos_initials = nn.ParameterList(
            #     nn.Parameter(torch.empty(self.pos_enc_dim, 1, device=self.device), requires_grad=not self.rand_pos_enc)
            #     for _ in range(self.num_initials)
            # )
            # for pos_initial in self.pos_initials:
            #     nn.init.normal_(pos_initial)

            self.pos_initials = nn.ParameterList(
                nn.Parameter(torch.empty(self.pos_enc_dim, 1, device=self.device), requires_grad=not self.rand_pos_enc and not self.rand_sketchy_pos_enc)
                for _ in range(self.n_gape)
            )
            for pos_initial in self.pos_initials:
                nn.init.normal_(pos_initial)

            # init transition weights
            shape = (self.pos_enc_dim,) if net_params['diag'] else (self.pos_enc_dim, self.pos_enc_dim)
            self.pos_transitions = nn.ParameterList(
                nn.Parameter(torch.Tensor(*shape), requires_grad=not self.rand_pos_enc and not self.rand_sketchy_pos_enc)
                for _ in range(self.n_gape)
            )
            for pos_transition in self.pos_transitions:
                if net_params['diag']:
                    nn.init.normal_(pos_transition)
                else:
                    nn.init.orthogonal_(pos_transition)
            # init linear layers for reshaping to hidden dim
            if self.gape_individual:
                self.embedding_pos_encs = nn.ModuleList(nn.Linear(self.pos_enc_dim, hidden_dim) for _ in range(self.n_gape))
            else:
                self.embedding_pos_encs = nn.ModuleList(nn.Linear(self.pos_enc_dim, hidden_dim) for _ in range(1))
            # self.embedding_pos_enc = nn.Linear(self.pos_enc_dim, hidden_dim)

            if self.n_gape > 1:
                self.gape_pool_vec = nn.Parameter(torch.Tensor(self.n_gape, 1), requires_grad=True)
                nn.init.normal_(self.gape_pool_vec)

            self.mat_pows = nn.ParameterList([nn.Parameter(torch.Tensor(size=(1,))) for _ in range(self.pow_of_mat)])
            for mat_pow in self.mat_pows:
                nn.init.constant_(mat_pow, 1)
            # self.adder = nn.Parameter(torch.Tensor(self.pos_enc_dim, 1), requires_grad=True)

        elif self.pagerank:
            self.embedding_pos_enc = nn.Linear(self.pos_enc_dim, hidden_dim)

        if self.rw_pos_enc:
            self.embedding_pos_enc = nn.Linear(self.pos_enc_dim, hidden_dim) 

        in_dim = 1
        if self.dataset in ("SBM_PATTERN", "MNIST", "CIFAR10", "cornell", "Cora"):
            in_dim = net_params['in_dim']
        # self.embedding_h = nn.Linear(in_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)

        self.use_pos_enc = self.pos_enc or self.wl_pos_enc or self.learned_pos_enc or self.rand_pos_enc or self.adj_enc or self.rw_pos_enc or self.rand_sketchy_pos_enc
        if self.use_pos_enc:
            self.logger.info(f"Using {self.pos_enc_dim} dimension positional encoding (# states if an automata enc, otherwise smallest k eigvecs)")

        if not self.use_pos_enc and self.dataset not in ('CYCLES', 'CIFAR10', 'MNIST', 'SBM_PATTERN', 'SBM_CLUSTER', 'Cora'):
            self.embedding_h = nn.Embedding(in_dim, hidden_dim)

        self.logger.info(f"Using matrix: {self.matrix_type}")
        self.logger.info(f"Matrix power: {self.pow_of_mat}")
        if self.power_method:
            self.logger.info(f"Using power method with {self.power_method_iters} iterations")
        
        self.transition_mul_mat = nn.Parameter(torch.Tensor(self.pos_enc_dim, self.pos_enc_dim), requires_grad=True)
        nn.init.normal_(self.transition_mul_mat)
        self.pos_adder = nn.Parameter(torch.Tensor(self.pos_enc_dim, 1), requires_grad=True)
        nn.init.normal_(self.pos_adder)

        self.eigen_bartels_stewart = net_params.get('eigen_bartels_stewart', False)

    def stack_strategy(self, num_nodes):
        """
            Given more than one initial weight vector, define the stack strategy.

            If n = number of nodes and k = number of weight vectors,
                by default, we repeat each initial weight vector n//k times
                and stack them together with final n-(n//k) weight vectors.
        """
        num_pos_initials = len(self.pos_initials)
        if num_pos_initials == 1:
            return torch.cat([self.pos_initials[0] for _ in range(num_nodes)], dim=1)

        remainder = num_nodes % num_pos_initials
        capacity = num_nodes - remainder
        out = torch.cat([self.pos_initials[i] for i in range(num_pos_initials)], dim=1)
        out = torch.repeat_interleave(out, capacity//num_pos_initials, dim=1)
        if remainder != 0:
            remaining_stack = torch.cat([self.pos_initials[-1] for _ in range(remainder)], dim=1)
            out = torch.cat([out, remaining_stack], dim=1)
        return out

    def kronecker(self, mat1, mat2):
        return torch.einsum('ab,cd->acbd', mat1, mat2).reshape(mat1.shape[0] * mat2.shape[0], mat1.shape[1] * mat2.shape[1])


    def sylvester(self, A, B, C, initial_vector):
        R, U = torch.linalg.eig(A)
        S, V = torch.linalg.eig(B)
        F = U.transpose(-1, -2) @ (C + 0j) @ V
        W = R[..., :, None] - S[..., None, :]
        # Y = F @ torch.linalg.inv(W)
        # Y = F @ torch.linalg.pinv(W)
        Y = F @ torch.linalg.pinv(W)

        # print("F: ", F.transpose(1, 0).shape)
        # print("torch.linalg.pinv(W): ", torch.linalg.pinv(W).transpose(1, 0).shape)

        return U @ Y @ initial_vector.type(torch.complex64) @ V.transpose(-1, -2)


    def learned_forward(self, g):
        if not self.diag:
            print("Must use diag with eigendecomposition-based Bartels-Stewart")
            exit()
        mat = self.type_of_matrix(g, self.matrix_type).to(self.device)
        vec_init = self.stack_strategy(g.number_of_nodes()).to(self.device)
        transition = torch.diag(self.pos_transitions[0])
        transition_inverse = torch.linalg.inv(transition).to(self.device)
        mat_product = transition_inverse @ vec_init
        pe = self.sylvester(transition_inverse, -mat, mat_product, vec_init)
        pe = pe.transpose(1, 0).type(torch.float32)
        pe = self.embedding_pos_encs[0](pe)
        if self.clamp:
            pe = torch.tanh(pe)
        return pe


    def forward(self, g, h, pos_enc=None, h_wl_pos_enc=None):
        if self.wl_pos_enc:
            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc) 
            h = h + h_wl_pos_enc
            return h

        # if not self.use_pos_enc and self.dataset in ("ZINC", "AQSOL", "SBM_PATTERN", "SBM_CLUSTER", "WikiCS", "cornell", "texas", "wisconsin", "Cora"):
        #     return h
        if not self.use_pos_enc:
            return h

        if not self.use_pos_enc:
            return self.embedding_h(h)

        # if self.rw_pos_enc:
        #     pe = self.rw_embedding_pos_enc(pos_enc)
        #     return pe

        pe = None

        if self.pos_enc or self.adj_enc or self.rw_pos_enc:
            pe = self.embedding_pos_enc(pos_enc)
        elif self.learned_pos_enc:
            if self.eigen_bartels_stewart:
                return self.learned_forward(g)

            mat = self.type_of_matrix(g, self.matrix_type)
            vec_init = self.stack_strategy(g.num_nodes())
            vec_init = vec_init.transpose(1, 0).flatten()
            kron_prod = torch.kron(mat.reshape(mat.shape[1], mat.shape[0]), self.pos_transitions[0]).to(self.device)

            if self.power_method:
                encs = vec_init
                lam = 0.25
                for _ in range(self.power_method_iters):
                    encs = ((lam*kron_prod) @ encs) + vec_init
                    # print(self.power_method_iters)
                    # norm = torch.norm(encs, p=2, dim=0)
                    # encs = encs / norm
                    # encs = encs.clamp(min=-1, max=1)
                    # print(encs)
            else:
                B = torch.eye(kron_prod.shape[1]).to(self.device) - kron_prod
                encs = torch.linalg.solve(B, vec_init)

            stacked_encs = torch.stack(encs.split(self.pos_enc_dim), dim=1)
            stacked_encs = stacked_encs.transpose(1, 0)
            pe = self.embedding_pos_encs[0](stacked_encs)

            if self.clamp:
                # pe = pe.clamp(-3, 3)
                pe = torch.tanh(pe)

            return pe
        elif self.rand_sketchy_pos_enc:
            mat = self.type_of_matrix(g, self.matrix_type)
            initial_vector = torch.cat([self.pos_initials[0] for _ in range(mat.shape[0])], dim=1)

            initial_adder = torch.cat([self.pos_adder for _ in range(mat.shape[0])], dim=1)
            initial_vector = (initial_vector + initial_adder).detach().cpu().numpy()

            self.pos_transitions[0] = self.pos_transitions[0] + self.transition_mul_mat
            transition_inv = torch.inverse(self.pos_transitions[0]).detach().cpu().numpy()
            mat_product = torch.inverse(self.pos_transitions[0]).detach().cpu() @ initial_vector
            # initial_vector = initial_vector.cpu().numpy()
            mat_product = mat_product.cpu().numpy()

            mat = mat.cpu().numpy()
            pe = scipy.linalg.solve_sylvester(transition_inv, -mat, mat_product)
            pe = torch.from_numpy(pe.T).float().to(self.device)
            return self.embedding_pos_encs[0](pe)

        elif self.rand_pos_enc:
            if self.power_method:
                mat = self.type_of_matrix(g, self.matrix_type)
                vec_init = self.stack_strategy(g.num_nodes())
                vec_init = vec_init.transpose(1, 0).flatten().to(self.device)
                kron_prod = torch.kron(mat.reshape(mat.shape[1], mat.shape[0]), self.pos_transitions[0]).to(self.device)
                pe = vec_init
                lam = 0.25
                for _ in range(self.power_method_iters):
                    pe = ((lam*kron_prod) @ pe) + vec_init
                    # norm = torch.norm(pe, p=2, dim=0)
                    # pe = pe / norm

                    # encs = encs.clamp(min=-1, max=1)
                pe = pe.reshape(self.pos_enc_dim, -1).transpose(1, 0).to(self.device)
            elif self.pow_of_mat > 1:
                device = torch.device("cpu")
                vec_init = self.stack_strategy(g.num_nodes())
                mat = self.type_of_matrix(g, self.matrix_type)
                transition_inv = torch.inverse(self.pos_transition).to(device)

                # AX + XB = Q
                #  X = alpha
                #  A = mu inverse
                #  B = -A
                #  Q = mu inverse + pi
                transition_inv = transition_inv.numpy()
                mat = mat.detach().cpu().numpy()
                vec_init = vec_init.cpu().numpy()
                pe = sp.linalg.solve_sylvester(transition_inv, -mat, transition_inv @ vec_init)
                pe = torch.from_numpy(pe.T).to(self.device)
            else:
                pe = pos_enc

            # if self.cat:
            #     return pe

            if self.n_gape > 1:
                pos_encs = [g.ndata[f'pos_enc_{i}'] for i in range(self.n_gape)]
                # if not self.cat:
                
                if self.gape_individual:
                    pos_encs = [self.embedding_pos_encs[i](pos_encs[i]) for i in range(self.n_gape)]

                if self.gape_softmax_before:
                    normalized_pos_encs = []
                    for pos_enc in pos_encs:
                        normalized_pos_enc = torch.softmax(pos_enc, dim=1)
                        normalized_pos_encs.append(normalized_pos_enc)
                    pos_encs = normalized_pos_encs

                pe = torch.stack(pos_encs, dim=0) # (n_gape, n_nodes, pos_enc_dim)

                pe = pe.permute(1, 2, 0) # (n_nodes, pos_enc_dim, n_gape)

                # pos_enc_block = self.embedding_pos_enc(pos_enc_block) # (n_gape, n_nodes, hidden_dim)

                # if self.gape_pooling == "mean":
                #     pos_enc_block = torch.mean(pos_enc_block, 0, keepdim=False) # (n_nodes, hidden_dim)
                # elif self.gape_pooling == 'sum':
                #     pos_enc_block = torch.sum(pos_enc_block, 0, keepdim=False)
                # elif self.gape_pooling == 'max':
                #     pos_enc_block = torch.max(pos_enc_block, 0, keepdim=False)[0]

                pe = pe @ self.gape_pool_vec

                if self.gape_softmax_after:
                    pe = torch.softmax(pe, dim=1)

                pe = pe.squeeze(2)
                
                if not self.gape_individual:
                    pe = self.embedding_pos_encs[0](pe)

                # pe = torch.softmax(pe, dim=1)
            else:
                if not self.cat:
                    pe = self.embedding_pos_encs[0](pe)

                if self.gape_softmax_after:
                    # pe = torch.softmax(pe, dim=0)
                    # pe = torch.relu(pe)
                    pe = torch.tanh(pe)

            # if self.dataset in ("CYCLES", "ZINC"):
            if self.clamp:
                # pe = pe.clamp(-3, 3)
                pe = torch.tanh(pe)
            return pe

        elif self.pagerank:
            graph = dgl.to_networkx(g.cpu())
            google_matrix = nx.google_matrix(graph).A
            pe = self.embedding_pos_enc(torch.from_numpy(google_matrix).to(self.device)[:, :self.pos_enc_dim].type(torch.float32))
            torch.save(pe, "/home/psoga/Documents/projects/benchmarking-gnns/google_matrix.pt")
        else:
            if self.dataset == "ZINC":
                pe = h
            elif self.dataset == "Cora":
                return h
            pe = self.embedding_h(h)
            # elif self.dataset == "CYCLES":
            #     pe = self.embedding_h(h)

        # # if self.dataset in ("CYCLES", "ZINC"):
        # #     return pe

        # return h + pe if pe is not None else h
        # if self.dataset in ("CYCLES", "ZINC", "AQSOL", "SBM_PATTERN", "SBM_CLUSTER", "WikiCS", "cornell", "texas", "Cora", "CSL"):
        #     return pe + h

        return pe

    def get_normalized_laplacian(self, g):
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        N = sp.sparse.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        L = sp.sparse.eye(g.number_of_nodes()) - N * A * N
        return L

    def type_of_matrix(self, g, matrix_type):
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
            matrix = torch.from_numpy(nx.laplacian_matrix(graph).A).to(self.device).type(torch.float32)
        elif matrix_type == "E":
            laplacian = self.get_normalized_laplacian(g)
            EigVal, EigVec = np.linalg.eig(laplacian.toarray())
            idx = EigVal.argsort() # increasing order
            EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
            matrix = torch.from_numpy(EigVec).float().to(self.device)

        # learnable adj matrix "masks"
        # matrix = torch.matrix_power(matrix, pow)

        if self.pow_of_mat > 1:
            matrices = [torch.matrix_power(matrix, i) for i in range(self.pow_of_mat)]

            # multiply each matrix by the corresponding mat_pow
            for i in range(len(matrices)):
                matrices[i] = matrices[i] * self.mat_pows[i]
            matrix = torch.sum(torch.stack(matrices, dim=0), dim=0)

        return matrix

