import os
from matplotlib import pyplot as plt
import torch
import dgl
import scipy
import pickle
import scipy.sparse as sp
import numpy as np
import torch.nn.functional as F
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})
from . import algos

def spd_encoding(g):
    # shortest_path_result, _ = algos.floyd_warshall(g.adj().to_dense().numpy().astype(int))
    # spatial_pos = torch.from_numpy((shortest_path_result)).long()
    
    # g = dgl.to_networkx(g)
    # shortest_paths = nx.floyd_warshall(g)
    # spatial_pos = [[-1]*g.number_of_nodes() for _ in range(g.number_of_nodes())]
    shortest_path_result, _ = algos.floyd_warshall(g.adjacency_matrix().to_dense().numpy().astype(int))
    spatial_pos = torch.from_numpy((shortest_path_result)).long()

    # for src, trg_dict in shortest_paths.items():
    #     for trg, distance in trg_dict.items():
    #         spatial_pos[src][trg] = distance
    #         spatial_pos[trg][src] = distance

    spatial_pos = torch.from_numpy(np.array(spatial_pos))
    # spatial_pos[spatial_pos == float('inf')] = 512
    spatial_pos = spatial_pos.type(torch.long)

    return spatial_pos

def add_spd_encodings(dataset):
    dataset.train.spatial_pos_lists = [spd_encoding(g) for g in dataset.train.graph_lists]
    dataset.val.spatial_pos_lists = [spd_encoding(g) for g in dataset.val.graph_lists]
    dataset.test.spatial_pos_lists = [spd_encoding(g) for g in dataset.test.graph_lists]
    return dataset

def simple_spectral_decomp(g):
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    EigVals, EigVecs = np.linalg.eigh(A.toarray())
    setattr(g, 'EigVecs', torch.from_numpy(EigVecs))
    setattr(g, 'EigVals', torch.from_numpy(EigVals))
    # g.ndata['EigVecs'] = torch.from_numpy(EigVecs).float()
    # g.ndata['EigVals'] = torch.from_numpy(EigVals).float()
    return g

def add_simple_spectral_decomp(dataset):
    dataset.train.graph_lists = [simple_spectral_decomp(g) for g in dataset.train.graph_lists]
    dataset.val.graph_lists = [simple_spectral_decomp(g) for g in dataset.val.graph_lists]
    dataset.test.graph_lists = [simple_spectral_decomp(g) for g in dataset.test.graph_lists]
    return dataset


def spectral_decomposition(g, pos_enc_dim):
    # Laplacian
    n = g.number_of_nodes()
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVals, EigVecs = np.linalg.eigh(L.toarray())
    EigVals, EigVecs = EigVals[: pos_enc_dim], EigVecs[:, :pos_enc_dim]  # Keep up to the maximum desired number of frequencies

    # Normalize and pad EigenVectors
    EigVecs = torch.from_numpy(EigVecs).float()
    EigVecs = F.normalize(EigVecs, p=2, dim=1, eps=1e-12, out=None)
    
    if n<pos_enc_dim:
        g.ndata['EigVecs'] = F.pad(EigVecs, (0, pos_enc_dim-n), value=float('nan'))
    else:
        g.ndata['EigVecs']= EigVecs
        
    
    #Save eigenvales and pad
    EigVals = torch.from_numpy(np.sort(np.abs(np.real(EigVals)))) #Abs value is taken because numpy sometimes computes the first eigenvalue approaching 0 from the negative
    
    if n<pos_enc_dim:
        EigVals = F.pad(EigVals, (0, pos_enc_dim-n), value=float('nan')).unsqueeze(0)
    else:
        EigVals=EigVals.unsqueeze(0)
        
    
    #Save EigVals node features
    g.ndata['EigVals'] = EigVals.repeat(g.number_of_nodes(),1).unsqueeze(2)
    
    return g


def add_spectral_decomposition(dataset, pos_enc_dim):
    dataset.train.graph_lists = [spectral_decomposition(g, pos_enc_dim) for g in dataset.train.graph_lists]
    dataset.val.graph_lists = [spectral_decomposition(g, pos_enc_dim) for g in dataset.val.graph_lists]
    dataset.test.graph_lists = [spectral_decomposition(g, pos_enc_dim) for g in dataset.test.graph_lists]
    return dataset

def random_walk_encoding(g, pos_enc_dim, type='partial', ret_pe=False):
    """
    Graph positional encoding w/ random walk
    """

    # Geometric diffusion features with Random Walk
    A = g.adjacency_matrix(scipy_fmt="csr")
    Dinv = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float) # D^-1
    RW = A * Dinv  
    M = RW

    nb_pos_enc = pos_enc_dim
    PE = [torch.from_numpy(M.diagonal()).float()]
    M_power = M
    for _ in range(nb_pos_enc-1):
        M_power = M_power * M
        if type == 'partial':
            PE.append(torch.from_numpy(M_power.diagonal()).float())
        else:
            PE.append(torch.from_numpy(M_power).float())
    PE = torch.stack(PE,dim=-1)
    if ret_pe:
        return PE

    g.ndata['pos_enc'] = PE  

    return g


def add_rw_pos_encodings(dataset, pos_enc_dim, type='partial'):
    dataset.train.graph_lists = [random_walk_encoding(g, pos_enc_dim, type) for g in dataset.train.graph_lists]
    dataset.val.graph_lists = [random_walk_encoding(g, pos_enc_dim, type) for g in dataset.val.graph_lists]
    dataset.test.graph_lists = [random_walk_encoding(g, pos_enc_dim, type) for g in dataset.test.graph_lists]
    return dataset


def multiple_automaton_encodings(g: dgl.DGLGraph, transition_matrix, initial_vector, diag=False, matrix='A', idx=0, model=None):
    pe = automaton_encoding(g, transition_matrix, initial_vector, diag, matrix, ret_pe=True, storage=None, idx=idx, model=model)
    key = f'pos_enc_{idx}'
    # if 'pos_enc' not in g.ndata:  
    g.ndata[key] = pe
    return g

def random_orientation(g: dgl.DGLGraph):
    edges = g.edges()
    src_tensor, dst_tensor = edges[0], edges[1]
    for i, (src, dst) in enumerate(zip(src_tensor, dst_tensor)):
        if (dst, src) in g.edges():
            p = np.random.rand()
            if p > 0.5:
                g.remove_edge(i)
    return g

def add_random_orientation(dataset):
    dataset.train.graph_lists = [random_orientation(g) for g in dataset.train.graph_lists]
    dataset.val.graph_lists = [random_orientation(g) for g in dataset.val.graph_lists]
    dataset.test.graph_lists = [random_orientation(g) for g in dataset.test.graph_lists]
    return dataset

def add_multiple_automaton_encodings(dataset, transition_matrices, initial_vectors, diag=False, matrix='A', model=None):
    transition_matrix = transition_matrices[0]
    initial_vector = initial_vectors[0]
    # for i, (_, _) in enumerate(zip(transition_matrices, initial_vectors)):
    for i, _ in enumerate(transition_matrices):
        dataset.train.graph_lists = [multiple_automaton_encodings(g, transition_matrix, initial_vector, diag, matrix, i, model) for g in dataset.train.graph_lists]
        dataset.val.graph_lists = [multiple_automaton_encodings(g, transition_matrix, initial_vector, diag, matrix, i, model) for g in dataset.val.graph_lists]
        dataset.test.graph_lists = [multiple_automaton_encodings(g, transition_matrix, initial_vector, diag, matrix, i, model) for g in dataset.test.graph_lists]

    # dump_encodings(dataset, transition_matrix.shape[0])
    return dataset

def automaton_encoding(g, transition_matrix, initial_vector, diag=False, matrix='A', ret_pe=False, storage=None, idx=0, model=None):
    # g = random_orientation(g)
    """
    Graph positional encoding w/ automaton weights
    """
    # transition_inv = transition_matrix.transpose(1, 0).cpu().numpy() # assuming the transition matrix is orthogonal

    # if diag:
    #     transition_matrix = torch.diag(transition_matrix)
        # torch.einsum('ij, kj -> ij', a, b) for matrix
        # torch.einsum('ij, j->ij', a, b) for vector
        # torch.einsum('ij, i->ij', a, b), a is a matrix, b is a vector
    # rw = random_walk_encoding(g, transition_matrix.shape[0], ret_pe=True)

    # A_hat = Dinv * A * Dinv

    # A = g.adjacency_matrix(scipy_fmt="csr")
    # Dinv = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(0) ** -1.0, dtype=float) # D^-1
    # alpha = 0.0001
    # PE = []
    # A_hat = A * Dinv
    # prev = A_hat

    # for _ in range(1, transition_matrix.shape[0]+1):
    #     pe_mat = alpha * np.linalg.inv((np.eye(g.number_of_nodes()) - (1-alpha) * A_hat))
    #     # A_hat = A_hat * prev

    #     pe = torch.from_numpy(pe_mat.diagonal()).float()
    #     PE.append(pe)

    # PE = torch.stack(PE,dim=-1).squeeze(0)

    # import seaborn as sb
    # import matplotlib.pyplot as plt

    # # PE -= PE.min(1, keepdim=True)[0]
    # # PE /= PE.max(1, keepdim=True)[0]

    # # rw -= rw.min(1, keepdim=True)[0]
    # # rw /= rw.max(1, keepdim=True)[0]

    # plt.figure("rw")
    # plt.title("rw")
    # ax1 = sb.heatmap(rw)
    # ax1.invert_yaxis()
    # plt.figure("ppr")
    # plt.title("ppr")
    # ax2 = sb.heatmap(PE)
    # ax2.invert_yaxis()
    # plt.show()    

    # exit()

    # if ret_pe:
    #     return PE

    # g.ndata['pos_enc'] = PE
    # return g

    transition_matrix = torch.nan_to_num(transition_matrix)
    if diag:
        transition_inv = transition_matrix**-1
    else:
        transition_inv = torch.linalg.inv(transition_matrix).cpu().numpy()

    if matrix == 'A':
        # Adjacency matrix
        mat = g.adjacency_matrix().to_dense().cpu().numpy()
        # mat = np.linalg.matrix_power(mat, idx+1)
    elif matrix == 'L':
        # Normalized Laplacian
        n = g.number_of_nodes()
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        D = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        L = sp.eye(n) - D * A * D
        mat = L.todense()
    elif matrix == 'SL':
        # Normalized unsigned Laplacian
        n = g.number_of_nodes()
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        D = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        L = sp.eye(n) + D * A * D
        mat = L.todense()
    elif matrix == 'UL':
        # Unnormalized Laplacian
        n = g.number_of_nodes()
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        D = sp.diags(dgl.backend.asnumpy(g.in_degrees()), dtype=float)
        mat = (A - D).todense()
    elif matrix == 'USL':
        # Unnormalized unsigned Laplacian
        n = g.number_of_nodes()
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        D = sp.diags(dgl.backend.asnumpy(g.in_degrees()), dtype=float)
        mat = (A + D).todense()
    elif matrix == 'E':
        # Laplacian eigenvector matrix
        n = g.number_of_nodes()
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        D = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        L = sp.eye(n) - D * A * D
        EigVal, EigVec = np.linalg.eig(L.todense())
        EigVec = EigVec[:, EigVal.argsort()] # increasing order
        mat = EigVec
    elif matrix == 'R':
        # Random walk matrix (1st power)
        n = g.number_of_nodes()
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        D = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float)
        mat = (A * D).todense()
    elif matrix == 'R2':
        # Random walk matrix (2nd power)
        n = g.number_of_nodes()
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        D = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float)
        mat = (A * D).todense()**2
    elif matrix == 'R20':
        # Random walk matrix (20th power)
        n = g.number_of_nodes()
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        D = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float)
        rw = (A*D).todense()
        m_power = rw

        for _ in range(20):
            m_power = m_power * rw
        mat = m_power

    elif matrix == 'RV':
        n = g.number_of_nodes()
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        D = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float)
        rw = (A*D).todense()
        m_power = rw
        # if idx > 0:
        #     l = idx * 2
        # else:
        #     l = 1

        for _ in range(idx+1):
            m_power = m_power * rw
        mat = m_power

    elif matrix == 'RWK':
        n = g.number_of_nodes()
        A = g.adjacency_matrix(scipy_fmt="csr")
        p_steps = n
        # p_steps = int(0.7*n)
        # p_steps = int(0.4*n)
        # p_steps = int(0.7*n)
        # p_steps = int(0.3*n)
        gamma = 1

        N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        I = sp.eye(n)
        L = I - N * A * N

        k_RW = I - gamma*L
        k_RW_power = k_RW
        for _ in range(p_steps - 1):
            k_RW_power = k_RW_power.dot(k_RW)

        k_RW_power = k_RW_power.toarray()
        mat = k_RW_power

    elif matrix == 'RWK16':
        n = g.number_of_nodes()
        A = g.adjacency_matrix(scipy_fmt="csr")
        p_steps = 16
        # p_steps = int(0.7*n)
        # p_steps = int(0.4*n)
        # p_steps = int(0.7*n)
        # p_steps = int(0.3*n)
        gamma = 1

        N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        I = sp.eye(n)
        L = I - N * A * N

        k_RW = I - gamma*L
        k_RW_power = k_RW
        for _ in range(p_steps - 1):
            k_RW_power = k_RW_power.dot(k_RW)

        k_RW_power = k_RW_power.toarray()
        mat = k_RW_power

    if model.pe_layer.gape_normalize_mat:
        D = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float) # D^-1
        mat = mat @ D
        # mat = D @ mat
        # mat = mat * 0.85
        # mat = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float) @ mat
        # A = g.adjacency_matrix(scipy_fmt="csr")
        # N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        # I = sp.eye(g.number_of_nodes())
        # L = I - N * A * N
        # mat = (N * A * N).todense()
        # mat = (I - N * A * N).todense()
        # k_RW = I - L
        # mat = k_RW.todense()

        # mat = mat * 0.9
    
    if model.pe_layer.ngape_betas:
        gape_beta = float(model.pe_layer.ngape_betas[idx])
    else:
        gape_beta = float(model.pe_layer.gape_beta)

    if model.pe_layer.gape_beta < 1:
        mat = mat * (1-gape_beta) # emulate pagerank


    # if model is None:
    # initial_vector = torch.cat([initial_vector for _ in range(mat.shape[0])], dim=1)
    # else: initial_vector = model.pe_layer.stack_strategy(g)

    # initial_vector = torch.fill(initial_vector, 1/g.number_of_nodes())
    if model.pe_layer.gape_weight_id:
        initial_vector = torch.zeros_like(initial_vector)
        # initial_vector.fill_diagonal_(1)
        import random
        rows, cols = initial_vector.shape
        indices = [n for n in range(rows)]
        for i in range(cols):
            p = random.choice(indices)
            initial_vector[p, i] = 1
    else:
        initial_vector = model.pe_layer.stack_strategy(g)

    if model.pe_layer.gape_beta:
        initial_vector = initial_vector * gape_beta # emulate pagerank

    # print(torch.linalg.svd(initial_vector)[1])
    initial_vector_torch = initial_vector.clone()
    # import random
    # if idx == 0:
    #     initial_vector = torch.cat([initial_vector for _ in range(mat.shape[0])], dim=1)
    # else:
    #     pi = torch.zeros(initial_vector.shape[0], g.number_of_nodes())
    #     index = random.randint(0, g.number_of_nodes()-1)
    #     pi[:, index] = initial_vector.squeeze(1)
    #     initial_vector = pi

    if diag:
        # mat_product = torch.einsum('ij, i->ij', initial_vector, transition_inv).cpu().numpy()
        mat_product = (torch.diag(transition_matrix) @ initial_vector).cpu().numpy() 
        transition_inv = torch.diag(transition_inv).cpu().numpy()
    else:
        initial_vector = initial_vector.cpu().numpy()
        mat_product = transition_inv @ initial_vector

    # initial_vector = torch.from_numpy(initial_vector).transpose(1, 0).flatten()
    # pe = initial_vector
    # lam = 0.25
    # mat = torch.from_numpy(mat)
    # for _ in range(1):
    #     kron_prod = torch.kron(mat.reshape(mat.shape[1], mat.shape[0]), transition_matrix).type(torch.float)
    #     pe = ((lam*kron_prod) @ pe) + initial_vector

    # pe = torch.stack((pe.split(transition_matrix.shape[0])), dim=1)
    # pe = pe.transpose(1, 0)

    pe = scipy.linalg.solve_sylvester(transition_inv, -mat, mat_product)
    pe = torch.from_numpy(pe.T).float()

    if storage is not None:
        storage['before']['mins'].append(torch.min(pe))
        storage['before']['maxs'].append(torch.max(pe))
        storage['before']['all'].extend(torch.flatten(pe).tolist())

    #     clameped_pe = torch.clamp(pe, -5, 5)
    #     # clameped_pe = 5*torch.tanh(pe)
    #     # clameped_pe = torch.nn.functional.normalize(pe, dim=1)
    #     # clameped_pe = torch.relu(pe)

    #     storage['after']['mins'].append(torch.min(clameped_pe))
    #     storage['after']['maxs'].append(torch.max(clameped_pe))
    #     storage['after']['all'].extend(torch.flatten(clameped_pe).tolist())

    #     return storage
    # pe = torch.clamp(pe, -6, 6)
    # pe = torch.nn.functional.normalize(pe, dim=1)
    # pe = torch.relu(pe)
    if model.pe_layer.gape_tau:
        pe = pe.to(torch.device('cpu'))
        initial_vector_torch = initial_vector_torch.to(torch.device('cpu'))
        pe = torch.mul(pe.T, initial_vector_torch).T

    if ret_pe:
        return pe

    g.ndata['pos_enc'] = pe
    return g

def add_automaton_encodings(dataset, transition_matrix, initial_vector, diag=False, matrix='A', model=None):
    # Graph positional encoding w/ pre-computed automaton encoding
    storage = {
        'before': {
            'mins': [],
            'maxs': [],
            'all': []
        },
        'after': {
            'mins': [],
            'maxs': [],
            'all': []
        }
    }

    # dataset.train.graph_lists = [automaton_encoding(g, transition_matrix, initial_vector, diag, matrix) for g in dataset.train.graph_lists]

    # for g in dataset.train.graph_lists:
    #     storage = automaton_encoding(g, transition_matrix, initial_vector, diag, 'A', False, storage)

    # # import matplotlib.pyplot as plt

    # print(f"max before: {max(storage['before']['all'])}")
    # print(f"min before: {min(storage['before']['all'])}")
    # print(f"variance before: {np.var(storage['before']['all'])}")
    # print(f"mean before: {np.mean(storage['before']['all'])}")

    # print(f"max after: {max(storage['after']['all'])}")
    # print(f"min after: {min(storage['after']['all'])}")
    # print(f"variance after: {np.var(storage['after']['all'])}")
    # print(f"mean after: {np.mean(storage['after']['all'])}")

    # plt.figure("Total tensor values before")
    # plt.hist(storage['before']['all'], bins=100)

    # plt.figure("Total tensor values after")
    # plt.hist(storage['after']['all'], bins=100)

    # train = []
    # val = []
    # test = []
    # for g in tqdm(dataset.train.graph_lists):
    #     train.append(automaton_encoding(g, transition_matrix, initial_vector, diag, matrix, False))
    # for g in tqdm(dataset.val.graph_lists):
    #     val.append(automaton_encoding(g, transition_matrix, initial_vector, diag, matrix, False))
    # for g in tqdm(dataset.test.graph_lists):
    #     test.append(automaton_encoding(g, transition_matrix, initial_vector, diag, matrix, False))
    
    # dataset.train.graph_lists = train
    # dataset.val.graph_lists = train
    # dataset.test.graph_lists = train
    dataset.train.graph_lists = [automaton_encoding(g, transition_matrix, initial_vector, diag, matrix, False, model=model) for g in dataset.train.graph_lists]
    dataset.val.graph_lists = [automaton_encoding(g, transition_matrix, initial_vector, diag, matrix, False, model=model) for g in dataset.val.graph_lists]
    dataset.test.graph_lists = [automaton_encoding(g, transition_matrix, initial_vector, diag, matrix, False, model=model) for g in dataset.test.graph_lists]
    # dump_encodings(dataset, transition_matrix.shape[0])
    return dataset

def add_random_walk_encoding_CSL(splits, pos_enc_dim):
    graphs = []
    for split in splits[0]:
        graphs.append(random_walk_encoding(split, pos_enc_dim))
    new_split = (graphs, splits[1])
    return new_split

def add_spd_encoding_CSL(splits):
    spatial_pos_list = []
    for split in splits[0]:
        spatial_pos_list.append(spd_encoding(split))
    new_split = (splits[0], splits[1], spatial_pos_list)
    return new_split

def add_spectral_decomposition_CSL(splits, pos_enc_dim):
    graphs = []
    for split in splits[0]:
        graphs.append(spectral_decomposition(split, pos_enc_dim))
    new_split = (graphs, splits[1])
    return new_split

def multiple_automaton_encodings_CSL(g, transition_matrix, initial_vector, idx=0, model=None):
    pe = automaton_encoding_CSL(g, transition_matrix, initial_vector, ret_pe=True, model=model)
    key = f'pos_enc_{idx}'
    # if 'pos_enc' not in g.ndata:
    g.ndata[key] = pe
    return g

def add_multiple_automaton_encodings_CSL(splits, model):
    transition_matrices = model.pe_layer.pos_transitions
    initial_vectors = model.pe_layer.pos_initials
    for i, (transition_matrix, initial_vector) in enumerate(zip(transition_matrices, initial_vectors)):
        # print(i)
        graphs = []
        for g in splits[0]:
            # initial_vector = model.pe_layer.stack_strategy(g.num_nodes())
            graphs.append(multiple_automaton_encodings_CSL(g, transition_matrix, initial_vector, idx=i, model=model))
        new_split = (graphs, splits[1])
    # dump_encodings(dataset, transition_matrix.shape[0])
    return new_split

def automaton_encoding_CSL(g, transition_matrix, initial_vector, ret_pe=False, idx=0, model=None, matrix_type='A', prev_graph=None):
    transition_inv = transition_matrix.transpose(1, 0).cpu().numpy() # assuming the transition matrix is orthogonal
    matrix = g.adjacency_matrix().to_dense().cpu().numpy()
    mat = matrix

    if matrix_type == 'A':
        # Adjacency matrix
        mat = g.adjacency_matrix().to_dense().cpu().numpy()
    elif matrix_type == 'L':
        # Normalized Laplacian
        n = g.number_of_nodes()
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        D = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        L = sp.eye(n) - D * A * D
        mat = L.todense()
    elif matrix_type == 'SL':
        # Normalized unsigned Laplacian
        n = g.number_of_nodes()
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        D = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        L = sp.eye(n) + D * A * D
        mat = L.todense()
    elif matrix_type == 'UL':
        # Unnormalized Laplacian
        n = g.number_of_nodes()
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        D = sp.diags(dgl.backend.asnumpy(g.in_degrees()), dtype=float)
        mat = (A - D).todense()
    elif matrix_type == 'USL':
        # Unnormalized unsigned Laplacian
        n = g.number_of_nodes()
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        D = sp.diags(dgl.backend.asnumpy(g.in_degrees()), dtype=float)
        mat = (A + D).todense()
    elif matrix_type == 'E':
        # Laplacian eigenvector matrix
        n = g.number_of_nodes()
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        D = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        L = sp.eye(n) - D * A * D
        EigVal, EigVec = np.linalg.eig(L.todense())
        EigVec = EigVec[:, EigVal.argsort()] # increasing order
        mat = EigVec
    elif matrix_type == 'R':
        # Random walk matrix (1st power)
        n = g.number_of_nodes()
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        D = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float)
        mat = (A * D).todense()
    elif matrix_type == 'R2':
        # Random walk matrix (2nd power)
        n = g.number_of_nodes()
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        D = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float)
        mat = (A * D).todense()**2
    elif matrix_type == 'R20':
        # Random walk matrix (20th power)
        n = g.number_of_nodes()
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        D = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float)
        rw = (A*D).todense()
        m_power = rw

        for _ in range(20):
            m_power = m_power * rw
        mat = m_power

    elif matrix_type == 'RV':
        n = g.number_of_nodes()
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        D = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float)
        rw = (A*D).todense()
        m_power = rw
        # if idx > 0:
        #     l = idx * 2
        # else:
        #     l = 1

        for _ in range(idx+1):
            m_power = m_power * rw
        mat = m_power

    elif matrix_type == 'RWK':
        n = g.number_of_nodes()
        A = g.adjacency_matrix(scipy_fmt="csr")
        p_steps = n
        # p_steps = int(0.7*n)
        # p_steps = int(0.4*n)
        # p_steps = int(0.7*n)
        # p_steps = int(0.3*n)
        gamma = 1

        N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        I = sp.eye(n)
        L = I - N * A * N

        k_RW = I - gamma*L
        k_RW_power = k_RW
        for _ in range(p_steps - 1):
            k_RW_power = k_RW_power.dot(k_RW)

        k_RW_power = k_RW_power.toarray()
        mat = k_RW_power

    elif matrix_type == 'RWK16':
        n = g.number_of_nodes()
        A = g.adjacency_matrix(scipy_fmt="csr")
        p_steps = 16
        # p_steps = int(0.7*n)
        # p_steps = int(0.4*n)
        # p_steps = int(0.7*n)
        # p_steps = int(0.3*n)
        gamma = 1

        N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        I = sp.eye(n)
        L = I - N * A * N

        k_RW = I - gamma*L
        k_RW_power = k_RW
        for _ in range(p_steps - 1):
            k_RW_power = k_RW_power.dot(k_RW)

        k_RW_power = k_RW_power.toarray()
        mat = k_RW_power

    if model.pe_layer.gape_normalize_mat:
        mat = mat @ sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float) # D^-1

    matrix = mat

    # if idx == 0:
    # initial_vector = torch.cat([initial_vector for _ in range(matrix.shape[0])], dim=1)
    initial_vector = model.pe_layer.stack_strategy(g.number_of_nodes())

    # initial_vector = torch.ones_like(initial_vector)
    # for i in range(initial_vector.shape[0]):
    #     initial_vector[:, i] = torch.full_like(initial_vector[:, i], i)

    # else:
    #     import random
    #     pi = torch.zeros(initial_vector.shape[0], g.number_of_nodes())
    #     index = random.randint(0, g.number_of_nodes()-1)
    #     pi[:, index] = initial_vector.squeeze(1)
    #     initial_vector = pi

    initial_vector = initial_vector.detach().cpu().numpy()
    pe = scipy.linalg.solve_sylvester(transition_inv, -matrix, transition_inv @ initial_vector)
    pe = torch.from_numpy(pe.T).float()

    # import seaborn as sb
    # import networkx as nx

    # def id_to_str(graph: nx.Graph):
    #     g = nx.Graph()
    #     g.add_edges_from([(str(edge[0]), str(edge[1])) for edge in graph.edges()])
    #     g.add_nodes_from([str(id) for id in graph.nodes()])
    #     return g

    # graph = id_to_str(dgl.to_networkx(g).to_undirected())

    # if prev_graph is not None:
    #     prev_graph = id_to_str(dgl.to_networkx(prev_graph).to_undirected())
    #     print(nx.isomorphism.is_isomorphic(prev_graph, graph))

    # k = len(min(nx.cycle_basis(graph), key=len))
    # n = g.number_of_nodes()
    # plt.title(f"{n}, {k}")
    # sb.heatmap(pe)

    # plt.show()
    # input()

    if ret_pe:
        return pe

    g.ndata['pos_enc'] = pe
    return g

def add_automaton_encodings_CSL(splits, model, matrix_type='A'):
    transition_matrix = model.pe_layer.pos_transitions[0]
    graphs = []
    prev_graph = None
    for i, split in enumerate(splits[0]):
        initial_vector = model.pe_layer.stack_strategy(split.num_nodes())
        # initial_vector = model.pe_layer.pos_initials[0]
        graphs.append(automaton_encoding_CSL(split, transition_matrix, initial_vector, False, i, model, matrix_type=matrix_type, prev_graph=prev_graph))
        prev_graph = split

    new_split = (graphs, splits[1])
    return new_split


def dump_encodings(dataset, pos_enc_dim):
    name = dataset.name
    if not os.path.exists(f'./{name}'):
        os.makedirs(f'./{name}')

    with open(f'./{name}/train_{pos_enc_dim}.pkl', 'wb+') as f:
        pickle.dump(dataset.train.graph_lists, f)

    with open(f'./{name}/val_{pos_enc_dim}.pkl', 'wb+') as f:
        pickle.dump(dataset.val.graph_lists, f)

    with open(f'./{name}/test_{pos_enc_dim}.pkl', 'wb+') as f:
        pickle.dump(dataset.test.graph_lists, f)


def load_encodings(dataset, pos_enc_dim):
    name = dataset.name
    with open(f'./{name}/train_{pos_enc_dim}.pkl', 'rb') as f:
        dataset.train.graph_lists = pickle.load(f)

    with open(f'./{name}/val_{pos_enc_dim}.pkl', 'rb') as f:
        dataset.val.graph_lists = pickle.load(f)

    with open(f'./{name}/test_{pos_enc_dim}.pkl', 'rb') as f:
        dataset.test.graph_lists = pickle.load(f)

    return dataset