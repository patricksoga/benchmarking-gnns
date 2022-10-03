import torch
import torch.nn as nn
import scipy as sp
import numpy as np
import networkx as nx
import dgl
import scipy
import time
from utils.main_utils import get_logger
from random import choices
import scipy.stats as sps

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
        self.diag = net_params.get('diag', False)
        self.matrix_type = net_params['matrix_type']
        self.logger = get_logger(net_params['log_file'])

        self.experiment_1 = net_params.get('experiment_1', False)
        self.gape_normalization = net_params.get('gape_normalization', 'none')
        self.gape_squash = net_params.get('gape_squash', 'none')
        self.seed_array = net_params['seed_array']

        self.power_method = net_params.get('power_method', False)
        self.power_method_iters = net_params.get('power_iters', 50)

        self.rand_sketchy_pos_enc = net_params.get('rand_sketchy_pos_enc', False)

        self.gape_norm = net_params.get('gape_norm', False)
        self.gape_div = net_params.get('gape_div', False)
        self.gape_weight_gen = net_params.get('gape_weight_gen', False)

        self.gape_symmetric = net_params.get('gape_symmetric', False)

        hidden_dim = net_params['hidden_dim']

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
                for _ in range(self.num_initials)
            )
            for pos_initial in self.pos_initials:
                nn.init.normal_(pos_initial)

            # init transition weights
            shape = (self.pos_enc_dim,) if net_params['diag'] else (self.pos_enc_dim, self.pos_enc_dim)

            # self.pos_transitions = nn.ParameterList(
            #     nn.Parameter(torch.Tensor(*shape), requires_grad=not self.rand_pos_enc and not self.rand_sketchy_pos_enc)
            #     for _ in range(self.n_gape)
            # )
            # for pos_transition in self.pos_transitions:
            #     if net_params['diag']:
            #         nn.init.normal_(pos_transition)
            #     else:
            #         nn.init.orthogonal_(pos_transition)
            #         if self.gape_div:
            #             pos_transition = pos_transition/self.pos_enc_dim
            #         elif self.gape_norm:
            #             pos_transition = pos_transition/torch.linalg.norm(pos_transition)

            shape = (int(self.pos_enc_dim*(self.pos_enc_dim + 1)/2),)

            transitions = [torch.empty(self.pos_enc_dim, self.pos_enc_dim) for _ in range(self.n_gape)]

            if self.gape_norm:
                normed_transitions = []
                for transition in transitions:
                    torch.nn.init.normal_(transition)
                    normed_transition = transition / torch.linalg.norm(transition)
                    normed_transitions.append(normed_transition)
                self.pos_transitions = nn.ParameterList(nn.Parameter(normed_transition, requires_grad=not self.rand_pos_enc and not self.rand_sketchy_pos_enc) for normed_transition in normed_transitions)
            # elif self.gape_symmetric and self.learned_pos_enc: # enforce symmetric transition matrix so that eigenvalues stay real
            #     self.pos_transitions = nn.ParameterList(
            #         nn.Parameter(torch.empty(self.pos_enc_dim-i) for i in range(self.pos_enc_dim))
            #     )
            #     for transition in self.pos_transitions:
            #         torch.nn.init.normal_(transition)

            elif self.gape_div:
                divd_tansitions = []
                for transition in transitions:
                    torch.nn.init.normal_(transition)
                    divd_transition = transition / torch.linalg.norm(transition)
                    divd_tansitions.append(divd_transition)
                self.pos_transitions = nn.ParameterList(nn.Parameter(divd_transition, requires_grad=not self.rand_pos_enc and not self.rand_sketchy_pos_enc) for divd_transition in divd_tansitions)
            elif self.gape_weight_gen:
                transition_matrices = []
                for transition in transitions:
                    random_matrix = torch.from_numpy(sps.ortho_group.rvs(self.pos_enc_dim))
                    random_eigvals = torch.rand(self.pos_enc_dim,)
                    random_matrix_inv = torch.linalg.inv(random_matrix)

                    transition_matrix = random_matrix @ torch.diag(random_eigvals).type(torch.double) @ random_matrix_inv
                    transition_matrices.append(transition_matrix)
                self.pos_transitions = nn.ParameterList(nn.Parameter(transition, requires_grad=not self.rand_pos_enc and not self.rand_sketchy_pos_enc) for transition in transition_matrices)
            else:
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
                    # nn.init.constant_(pos_transition, 50)

            # for transition in transitions:
            #     values = torch.zeros(*shape)
            #     torch.nn.init.normal_(values)

            #     i, j = torch.triu_indices(self.pos_enc_dim, self.pos_enc_dim)
            #     symm_transition = transition.detach()
            #     symm_transition[i, j] = values
            #     symm_transition.T[i, j] = values
            #     symm_transition = symm_transition / torch.linalg.norm(symm_transition)
            #     symm_transitions.append(symm_transition)
            
            # self.pos_transitions = nn.ParameterList(nn.Parameter(symm_transition, requires_grad=not self.rand_pos_enc and not self.rand_sketchy_pos_enc) for symm_transition in symm_transitions)

            # self.pos_transitions = nn.ParameterList(
            #     nn.Parameter(torch.empty(*shape), requires_grad=not self.rand_pos_enc and not self.rand_sketchy_pos_enc)
            #     for _ in range(self.n_gape)
            # )
            
            # for pos_transition in self.pos_transitions:
            #     i, j = torch.triu_indices(self.pos_enc_dim, self.pos_enc_dim)
            #     A = torch.zeros(self.pos_enc_dim, self.pos_enc_dim)
            #     torch.nn.init.normal_(pos_transition)
            #     values = pos_transition.clone()
            #     A[i, j] = values
            #     A.T[i, j] = values
            #     pos_transition = A

            # init linear layers for reshaping to hidden dim
            if self.gape_individual:
                self.embedding_pos_encs = nn.ModuleList(nn.Linear(self.pos_enc_dim, hidden_dim) for _ in range(self.n_gape))
            else:
                self.embedding_pos_encs = nn.ModuleList(nn.Linear(self.pos_enc_dim, hidden_dim) for _ in range(1))
            # self.embedding_pos_enc = nn.Linear(self.pos_enc_dim, hidden_dim)

            if self.n_gape > 1:
                self.gape_pool_vec = nn.Parameter(torch.Tensor(self.n_gape, 1), requires_grad=True)
                nn.init.normal_(self.gape_pool_vec)

            # self.mat_pows = nn.ParameterList([nn.Parameter(torch.Tensor(size=(1,))) for _ in range(self.pow_of_mat)])
            # for mat_pow in self.mat_pows:
            #     nn.init.constant_(mat_pow, 1)

        elif self.pagerank:
            self.embedding_pos_enc = nn.Linear(self.pos_enc_dim, hidden_dim)

        if self.rw_pos_enc:
            self.embedding_pos_enc = nn.Linear(self.pos_enc_dim, hidden_dim) 

        self.use_pos_enc = self.pos_enc or self.learned_pos_enc or self.rand_pos_enc or self.adj_enc or self.rw_pos_enc or self.rand_sketchy_pos_enc
        if self.use_pos_enc:
            self.logger.info(f"Using {self.pos_enc_dim} dimension positional encoding (# states if an automata enc, otherwise smallest k eigvecs)")

        # if not self.use_pos_enc and self.dataset not in ('CYCLES', 'CIFAR10', 'MNIST', 'SBM_PATTERN', 'SBM_CLUSTER', 'Cora'):
        #     self.embedding_h = nn.Embedding(in_dim, hidden_dim)

        self.logger.info(f"Using matrix: {self.matrix_type}")
        self.logger.info(f"Matrix power: {self.pow_of_mat}")
        if self.power_method:
            self.logger.info(f"Using power method with {self.power_method_iters} iterations")
        
        self.transition_mul_mat = nn.Parameter(torch.Tensor(self.pos_enc_dim, self.pos_enc_dim), requires_grad=True)
        nn.init.normal_(self.transition_mul_mat)
        self.pos_adder = nn.Parameter(torch.Tensor(self.pos_enc_dim, 1), requires_grad=True)
        nn.init.normal_(self.pos_adder)

        self.eigen_bartels_stewart = net_params.get('eigen_bartels_stewart', False)
        if self.eigen_bartels_stewart:
            self.pos_transition_inv = nn.Parameter(torch.linalg.inv(self.pos_transitions[0]))

    def stack_strategy(self, num_nodes):
        num_pos_initials = len(self.pos_initials)
        try:
            num_nodes = num_nodes.number_of_nodes()
        except: pass

        options = [i for i in range(num_pos_initials)]
        indices = choices(options, k=num_nodes)
        out = torch.cat([self.pos_initials[i] for i in indices], dim=1)
        return out
        """
            Given more than one initial weight vector, define the stack strategy.

            If n = number of nodes and k = number of weight vectors,
                by default, we repeat each initial weight vector n//k times
                and stack them together with final n-(n//k) weight vectors.
        """
        try:
            num_nodes = num_nodes.number_of_nodes()
        except: pass
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


    def sylvester(self, A, B, C):
        m = B.shape[-1]
        n = A.shape[-1]
        R, U = torch.linalg.eig(A)
        S, V = torch.linalg.eig(B)
        # S, V = B
        # t1 = time.time()
        F = torch.linalg.solve(U, (C + 0j) @ V)
        # print(time.time()-t1)
        W = R[..., :, None] - S[..., None, :]
        Y = F / W
        X = U[...,:n,:n] @ Y[...,:n,:m] @ torch.linalg.inv(V)[...,:m,:m]
        return X

    def learned_forward(self, g, graph_spectra):
        # if not self.diag:
        #     print("Must use diag with eigendecomposition-based Bartels-Stewart")
        #     exit()
        mat = self.type_of_matrix(g, self.matrix_type).to(self.device)
        vec_init = self.stack_strategy(g.number_of_nodes()).to(self.device)
        # transition = torch.diag(self.pos_transitions[0])

        if self.gape_symmetric:
            triu = torch.triu(self.pos_transitions[0])
            values = triu[triu != 0]
            i, j = torch.triu_indices(self.pos_enc_dim, self.pos_enc_dim)
            triu[i, j] = values
            triu.T[i, j] = values
            transition = triu
        else:
            transition = self.pos_transitions[0]

        transition_inverse = torch.linalg.inv(transition).to(self.device)
        mat_product = transition_inverse @ vec_init
        # mat_product = self.pos_transition_inv @ vec_init
        pe = self.sylvester(transition_inverse, -mat, mat_product)
        # eigvals, eigvecs = g.EigVals, g.EigVecs
        # pe = self.sylvester(self.pos_transition_inv, (eigvals, eigvecs), mat_product)
        pe = pe.transpose(1, 0).type(torch.float32)
        pe = self.embedding_pos_encs[0](pe)
        if self.clamp:
            pe = torch.tanh(pe)
        return pe


    def forward(self, g, h, pos_enc=None, graph_spectra=None):
        pe = pos_enc
        if not self.use_pos_enc:
            return h

        if self.rw_pos_enc:
            pe = self.embedding_pos_enc(pos_enc)
            return pe

        if self.pos_enc or self.adj_enc or self.rw_pos_enc:
            pe = self.embedding_pos_enc(pos_enc)
        elif self.learned_pos_enc:
            if self.eigen_bartels_stewart:
                return self.learned_forward(g, graph_spectra)

            mat = self.type_of_matrix(g, self.matrix_type)
            vec_init = self.stack_strategy(g.num_nodes())
            vec_init = vec_init.transpose(1, 0).flatten()
            kron_prod = torch.kron(mat.reshape(mat.shape[1], mat.shape[0]), self.pos_transitions[0]).to(self.device)

            if self.power_method:
                encs = vec_init
                lam = 0.25
                for _ in range(self.power_method_iters):
                    encs = ((lam*kron_prod) @ encs) + vec_init
                    # norm = torch.norm(encs, p=2, dim=0)
                    # encs = encs / norm
                    # encs = encs.clamp(min=-1, max=1)
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
                transition_inv = transition_inv.numpy()
                mat = mat.detach().cpu().numpy()
                vec_init = vec_init.cpu().numpy()
                pe = sp.linalg.solve_sylvester(transition_inv, -mat, transition_inv @ vec_init)
                pe = torch.from_numpy(pe.T).to(self.device)
            else:
                pe = pos_enc

            if self.n_gape > 1:
                pos_encs = [g.ndata[f'pos_enc_{i}'] for i in range(self.n_gape)]
                
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

            else:

                # experimenting with normalization/squashing
                # pre_modified = pe
                # if self.gape_squash == 'softplus':
                #     pe = torch.nn.functional.softplus(pe)
                # if self.gape_squash == 'exp':
                #     pe = torch.exp(pe)
                # if self.gape_squash == 'square':
                #     pe = torch.mul(pe, pe)
                # if self.gape_squash == 'tanh':
                #     pe = torch.tanh(pe)

                # if self.gape_normalization == 'max':
                #     pe /= pe.max()
                # if self.gape_normalization == 'softmax_0':
                #     pe = torch.softmax(pe, dim=0)
                # if self.gape_normalization == 'softmax_1':
                #     pe = torch.softmax(pe, dim=1)

                # if self.experiment_1:
                #     try:
                #         pes = torch.load(f'./data/{self.dataset}_{self.gape_squash}_{self.gape_normalization}_{self.seed_array[0]}.pt')
                #         pes.append((pe, pre_modified))
                #     except:
                #         pes = [(pe, pre_modified)]
                #     torch.save(pes, f'./data/{self.dataset}_{self.gape_squash}_{self.gape_normalization}_{self.seed_array[0]}.pt')

                if not self.cat:
                    pe = self.embedding_pos_encs[0](pos_enc)

                if self.gape_softmax_after:
                    pe = torch.tanh(pe)

            if self.clamp:
                pe = torch.tanh(pe)

            pre_modified = pe

            if self.gape_squash == 'softplus':
                pe = torch.nn.functional.softplus(pe)
            if self.gape_squash == 'exp':
                pe = torch.exp(pe)
            if self.gape_squash == 'square':
                pe = torch.mul(pe, pe)
            if self.gape_squash == 'tanh':
                pe = torch.tanh(pe)

            if self.gape_normalization == 'max':
                pe = pe / pe.max()
            if self.gape_normalization == 'softmax_0':
                pe = torch.softmax(pe, dim=0)
            if self.gape_normalization == 'softmax_1':
                pe = torch.softmax(pe, dim=1)

            if self.experiment_1:
                try:
                    pes = torch.load(f'./data/{self.dataset}_{self.gape_squash}_{self.gape_normalization}_{self.seed_array[0]}.pt')
                    pes.append((pe, pre_modified, g))
                except:
                    pes = [(pe, pre_modified, g)]
                torch.save(pes, f'./data/{self.dataset}_{self.gape_squash}_{self.gape_normalization}_{self.seed_array[0]}.pt')

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

        # pe = torch.dropout(pe, p=0.05)
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

