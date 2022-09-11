import os
import torch
import dgl
import scipy
import pickle
import scipy.sparse as sp
import numpy as np
import torch.nn.functional as F
import pyximport
import networkx as nx

pyximport.install(setup_args={"include_dirs": np.get_include()})
from . import algos

def spd_encoding(g: dgl.DGLGraph):
    # shortest_path_result, _ = algos.floyd_warshall(g.adj().to_dense().numpy().astype(int))
    # spatial_pos = torch.from_numpy((shortest_path_result)).long()
    
    g = dgl.to_networkx(g)
    shortest_paths = nx.floyd_warshall(g)
    spatial_pos = [[-1]*g.number_of_nodes() for _ in range(g.number_of_nodes())]

    for src, trg_dict in shortest_paths.items():
        for trg, distance in trg_dict.items():
            spatial_pos[src][trg] = distance
            spatial_pos[trg][src] = distance

    spatial_pos = torch.from_numpy(np.array(spatial_pos))
    # spatial_pos[spatial_pos == float('inf')] = 512
    spatial_pos = spatial_pos.type(torch.long)

    return spatial_pos

def add_spd_encodings(dataset):
    dataset.train.spatial_pos_lists = [spd_encoding(g) for g in dataset.train.graph_lists]
    dataset.val.spatial_pos_lists = [spd_encoding(g) for g in dataset.val.graph_lists]
    dataset.test.spatial_pos_lists = [spd_encoding(g) for g in dataset.test.graph_lists]
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

def random_walk_encoding(g, pos_enc_dim, type='partial'):
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
    g.ndata['pos_enc'] = PE  

    return g


def add_rw_pos_encodings(dataset, pos_enc_dim, type='partial'):
    dataset.train.graph_lists = [random_walk_encoding(g, pos_enc_dim, type) for g in dataset.train.graph_lists]
    dataset.val.graph_lists = [random_walk_encoding(g, pos_enc_dim, type) for g in dataset.val.graph_lists]
    dataset.test.graph_lists = [random_walk_encoding(g, pos_enc_dim, type) for g in dataset.test.graph_lists]
    return dataset


def multiple_automaton_encodings(g: dgl.DGLGraph, transition_matrix, initial_vector, diag=False, matrix='A', idx=0):
    pe = automaton_encoding(g, transition_matrix, initial_vector, diag, matrix, ret_pe=True, storage=None, idx=idx)
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

def add_multiple_automaton_encodings(dataset, transition_matrices, initial_vectors, diag=False, matrix='A'):
    for i, (transition_matrix, initial_vector) in enumerate(zip(transition_matrices, initial_vectors)):
        dataset.train.graph_lists = [multiple_automaton_encodings(g, transition_matrix, initial_vector, diag, matrix, i) for g in dataset.train.graph_lists]
        dataset.val.graph_lists = [multiple_automaton_encodings(g, transition_matrix, initial_vector, diag, matrix, i) for g in dataset.val.graph_lists]
        dataset.test.graph_lists = [multiple_automaton_encodings(g, transition_matrix, initial_vector, diag, matrix, i) for g in dataset.test.graph_lists]

    # dump_encodings(dataset, transition_matrix.shape[0])
    return dataset

def automaton_encoding(g, transition_matrix, initial_vector, diag=False, matrix='A', ret_pe=False, storage=None, idx=0):
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

    if diag:
        transition_inv = transition_matrix**-1
    else:
        transition_inv = torch.inverse(transition_matrix).cpu().numpy()

    if matrix == 'A':
        # Adjacency matrix
        mat = g.adjacency_matrix().to_dense().cpu().numpy()
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
    elif matrix == 'RWK':
        n = g.number_of_nodes()
        A = g.adjacency_matrix(scipy_fmt="csr")
        # p_steps = int(0.5*n)
        p_steps = n
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

    initial_vector = torch.cat([initial_vector for _ in range(mat.shape[0])], dim=1)
    # if idx == 0:
    #     initial_vector = torch.cat([initial_vector for _ in range(mat.shape[0])], dim=1)
    # else:
    #     pi = torch.zeros(initial_vector.shape[0], g.number_of_nodes())
    #     index = random.randint(0, g.number_of_nodes()-1)
    #     pi[:, index] = initial_vector.squeeze(1)
    #     initial_vector = pi

    if diag:
        mat_product = torch.einsum('ij, i->ij', initial_vector, transition_inv).cpu().numpy()
        transition_inv = torch.diag(transition_inv).cpu().numpy()
    else:
        initial_vector = initial_vector.cpu().numpy()
        mat_product = transition_inv @ initial_vector

    pe = scipy.linalg.solve_sylvester(transition_inv, -mat, mat_product)
    pe = torch.from_numpy(pe.T).float()

    # if storage is not None:
    #     storage['mins'].append(torch.min(pe))
    #     storage['maxs'].append(torch.max(pe))
    #     storage['all'].extend(torch.flatten(pe).tolist())
    #     return storage
    pe = torch.clamp(pe, -5, 5)
    # pe = torch.tanh(pe)

    if ret_pe:
        return pe

    g.ndata['pos_enc'] = pe
    return g

def add_automaton_encodings(dataset, transition_matrix, initial_vector, diag=False, matrix='A'):
    # Graph positional encoding w/ pre-computed automaton encoding
    storage = {
        'mins': [],
        'maxs': [],
        'all': []
    }
    # dataset.train.graph_lists = [automaton_encoding(g, transition_matrix, initial_vector, diag, matrix) for g in dataset.train.graph_lists]
    for g in dataset.train.graph_lists:
        storage = automaton_encoding(g, transition_matrix, initial_vector, diag, matrix, False, storage)

    # print(f"max: {max(storage['all'])}")
    # print(f"min: {min(storage['all'])}")

    # import matplotlib.pyplot as plt
    # plt.figure("Max tensor values")
    # plt.hist(storage['maxs'], bins=10)

    # plt.figure("Min tensor values")
    # plt.hist(storage['mins'], bins=10)

    # plt.figure("Total tensor values")
    # plt.hist(storage['all'], bins=10)

    # plt.show()

    dataset.train.graph_lists = [automaton_encoding(g, transition_matrix, initial_vector, diag, matrix, False) for g in dataset.train.graph_lists]
    dataset.val.graph_lists = [automaton_encoding(g, transition_matrix, initial_vector, diag, matrix, False) for g in dataset.val.graph_lists]
    dataset.test.graph_lists = [automaton_encoding(g, transition_matrix, initial_vector, diag, matrix, False) for g in dataset.test.graph_lists]
    # dump_encodings(dataset, transition_matrix.shape[0])
    return dataset


def add_spd_encoding_CSL(splits):
    graphs = []
    for split in splits[0]:
        graphs.append(spd_encoding(split))
    new_split = (graphs, splits[1])
    return new_split

def multiple_automaton_encodings_CSL(g, transition_matrix, initial_vector, idx=0):
    pe = automaton_encoding_CSL(g, transition_matrix, initial_vector, ret_pe=True)
    key = f'pos_enc_{idx}'
    # if 'pos_enc' not in g.ndata:
    g.ndata[key] = pe
    return g

def add_multiple_automaton_encodings_CSL(splits, model):
    transition_matrices = model.pe_layer.pos_transitions
    initial_vectors = model.pe_layer.pos_initials
    for i, (transition_matrix, initial_vector) in enumerate(zip(transition_matrices, initial_vectors)):
        graphs = []
        for split in splits[0]:
            initial_vector = model.pe_layer.stack_strategy(split.num_nodes())
            graphs.append(multiple_automaton_encodings_CSL(split, transition_matrix, initial_vector, idx=i))
        new_split = (graphs, splits[1])
    # dump_encodings(dataset, transition_matrix.shape[0])
    return new_split

def automaton_encoding_CSL(g, transition_matrix, initial_vector, ret_pe=False, idx=0):
    transition_inv = transition_matrix.transpose(1, 0).cpu().numpy() # assuming the transition matrix is orthogonal
    matrix = g.adjacency_matrix().to_dense().cpu().numpy()

    if idx == 0:
        initial_vector = torch.cat([initial_vector for _ in range(matrix.shape[0])], dim=1)
    else:
        import random
        pi = torch.zeros(initial_vector.shape[0], g.number_of_nodes())
        index = random.randint(0, g.number_of_nodes()-1)
        pi[:, index] = initial_vector.squeeze(1)
        initial_vector = pi

    initial_vector = initial_vector.cpu().numpy()
    pe = scipy.linalg.solve_sylvester(transition_inv, -matrix, transition_inv @ initial_vector)
    pe = torch.from_numpy(pe.T).float()

    if ret_pe:
        return pe

    g.ndata['pos_enc'] = pe
    return g

def add_automaton_encodings_CSL(splits, model):
    transition_matrix = model.pe_layer.pos_transitions[0]
    graphs = []
    for i, split in enumerate(splits[0]):
        # initial_vector = model.pe_layer.stack_strategy(split.num_nodes())
        initial_vector = model.pe_layer.pos_initials[0]
        graphs.append(automaton_encoding_CSL(split, transition_matrix, initial_vector, False, i))

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