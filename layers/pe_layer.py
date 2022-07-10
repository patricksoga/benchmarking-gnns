import torch
import torch.nn as nn

class PELayer(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.device = net_params['device']
        self.pos_enc = net_params.get('pos_enc', False)
        self.learned_pos_enc = net_params.get('learned_pos_enc', False)
        self.rand_pos_enc = net_params.get('rand_pos_enc', False)
        self.pos_enc_dim = net_params.get('pos_enc_dim', 0)
        self.wl_pos_enc = net_params.get('wl_pos_enc', False)
        self.dataset = net_params['dataset']

        hidden_dim = net_params['hidden_dim']
        max_wl_role_index = 37 # this is maximum graph size in the dataset

        if self.pos_enc:
            print("Using Laplacian position encoding")
            self.embedding_pos_enc = nn.Linear(self.pos_enc_dim, hidden_dim)
        elif self.rand_pos_enc:
            print("Using random automata position encoding")
            self.pos_initial = nn.Parameter(torch.Tensor(self.pos_enc_dim, 1), requires_grad=False)
            self.pos_transition = nn.Parameter(torch.Tensor(self.pos_enc_dim, self.pos_enc_dim), requires_grad=False)
            nn.init.normal_(self.pos_initial)
            nn.init.orthogonal_(self.pos_transition)
            self.embedding_pos_enc = nn.Linear(self.pos_enc_dim, hidden_dim)
        elif self.learned_pos_enc:
            print("Using learned automata position encoding")
            self.pos_initial = nn.Parameter(torch.Tensor(self.pos_enc_dim, 1))
            self.pos_transition = nn.Parameter(torch.Tensor(self.pos_enc_dim, self.pos_enc_dim))
            nn.init.normal_(self.pos_initial)
            nn.init.orthogonal_(self.pos_transition)
            self.embedding_pos_enc = nn.Linear(self.pos_enc_dim, hidden_dim)
        in_dim = 1
        if self.dataset == "SBM_PATTERN":
            in_dim = net_params['in_dim']
        self.embedding_h = nn.Linear(in_dim, hidden_dim)

        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)

        self.use_pos_enc = self.pos_enc or self.wl_pos_enc or self.learned_pos_enc or self.rand_pos_enc
        if self.use_pos_enc:
            print(f"Using {self.pos_enc_dim} dimension positional encoding")

    def forward(self, g, h, pos_enc=None, h_wl_pos_enc=None):
        if self.wl_pos_enc:
            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc) 
            h = h + h_wl_pos_enc
            return h

        # if self.pos_enc:
        #     h += self.embedding_pos_enc(pos_enc)
        #     return h
        # elif self.learned_pos_enc or self.rand_pos_enc:
        #     A = g.adjacency_matrix().to_dense().to(self.device)
        #     z = torch.zeros(self.pos_enc_dim, g.num_nodes()-1, requires_grad=False).to(self.device)
        #     vec_init = torch.cat((self.pos_initial, z), dim=1).to(self.device)
        #     vec_init = vec_init.transpose(1, 0).flatten()
        #     kron_prod = torch.kron(A.t().contiguous(), self.pos_transition).to(self.device)
        #     B = torch.eye(kron_prod.shape[1]).to(self.device) - kron_prod
        #     encs = torch.linalg.solve(B, vec_init)
        #     stacked_encs = torch.stack(encs.split(self.pos_enc_dim), dim=1).transpose(1, 0)
        #     h += self.embedding_pos_enc(stacked_encs)
        #     return h
        # else:
        #     if self.dataset == "ZINC":
        #         return h
        #     if self.dataset == "CYCLES":
        #         h = self.embedding_h(h)
        #     return h
        pe = None
        if self.pos_enc:
            pe = self.embedding_pos_enc(pos_enc)
            print(pe.shape)
        elif self.learned_pos_enc or self.rand_pos_enc:
            A = g.adjacency_matrix().to_dense().to(self.device)
            z = torch.zeros(self.pos_enc_dim, g.num_nodes()-1, requires_grad=False).to(self.device)
            vec_init = torch.cat((self.pos_initial, z), dim=1).to(self.device)
            vec_init = vec_init.transpose(1, 0).flatten()
            kron_prod = torch.kron(A.t().contiguous(), self.pos_transition).to(self.device)
            B = torch.eye(kron_prod.shape[1]).to(self.device) - kron_prod
            encs = torch.linalg.solve(B, vec_init)
            stacked_encs = torch.stack(encs.split(self.pos_enc_dim), dim=1)
            print(stacked_encs.shape)
            stacked_encs = stacked_encs.transpose(1, 0)
            pe = self.embedding_pos_enc(stacked_encs)
            print('h shape: ', h.shape)
        else:
            if self.dataset == "ZINC":
                pe = h
            elif self.dataset == "CYCLES":
                pe = self.embedding_h(h)
        
        if self.dataset in ("CYCLES", "ZINC"):
            return pe
        return h + pe if pe is not None else h