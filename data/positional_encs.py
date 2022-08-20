import os
import torch
import dgl
import scipy
import pickle
import scipy.sparse as sp

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

def automaton_encoding(g, transition_matrix, initial_vector, diag=False, matrix='A'):
    """
    Graph positional encoding w/ automaton weights
    """
    # transition_inv = transition_matrix.transpose(1, 0).cpu().numpy() # assuming the transition matrix is orthogonal

    if diag:
        transition_matrix = torch.diag(transition_matrix)

    transition_inv = torch.inverse(transition_matrix).cpu().numpy()
    if matrix == 'A':
        mat = g.adjacency_matrix().to_dense().cpu().numpy()
    elif matrix == 'L':
        n = g.number_of_nodes()
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        L = sp.eye(n) - N * A * N
        mat = L.todense()
    elif matrix == 'UL':
        n = g.number_of_nodes()
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        N = sp.diags(dgl.backend.asnumpy(g.in_degrees()), dtype=float)
        L = A - N

    initial_vector = torch.cat([initial_vector for _ in range(mat.shape[0])], dim=1)
    initial_vector = initial_vector.cpu().numpy()
    pe = scipy.linalg.solve_sylvester(transition_inv, -mat, transition_inv @ initial_vector)
    g.ndata['pos_enc'] = torch.from_numpy(pe.T).float()
    return g

def add_automaton_encodings(dataset, transition_matrix, initial_vector, diag=False, matrix='A'):
    # Graph positional encoding w/ pre-computed automaton encoding
    dataset.train.graph_lists = [automaton_encoding(g, transition_matrix, initial_vector, diag, matrix) for g in dataset.train.graph_lists]
    dataset.val.graph_lists = [automaton_encoding(g, transition_matrix, initial_vector, diag, matrix) for g in dataset.val.graph_lists]
    dataset.test.graph_lists = [automaton_encoding(g, transition_matrix, initial_vector, diag, matrix) for g in dataset.test.graph_lists]
    # dump_encodings(dataset, transition_matrix.shape[0])
    return dataset


def automaton_encoding_CSL(g, transition_matrix, initial_vector):
    transition_inv = transition_matrix.transpose(1, 0).cpu().numpy() # assuming the transition matrix is orthogonal
    matrix = g.adjacency_matrix().to_dense().cpu().numpy()
    initial_vector = initial_vector.cpu().numpy()
    pe = scipy.linalg.solve_sylvester(transition_inv, -matrix, transition_inv @ initial_vector)
    g.ndata['pos_enc'] = torch.from_numpy(pe.T).float()
    return g

def add_automaton_encodings_CSL(splits, model):
    transition_matrix = model.pe_layer.pos_transition
    graphs = []
    for split in splits[0]:
        initial_vector = model.pe_layer.stack_strategy(split.num_nodes())
        graphs.append(automaton_encoding_CSL(split, transition_matrix, initial_vector))

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