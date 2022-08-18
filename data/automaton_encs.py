import os
import torch
import scipy
import pickle

def automaton_encoding(g, transition_matrix, initial_vector):
    """
    Graph positional encoding w/ automaton weights
    """
    transition_inv = transition_matrix.transpose(1, 0).cpu().numpy() # assuming the transition matrix is orthogonal
    matrix = g.adjacency_matrix().to_dense().cpu().numpy()
    initial_vector = torch.cat([initial_vector for _ in range(matrix.shape[0])], dim=1)
    initial_vector = initial_vector.cpu().numpy()
    pe = scipy.linalg.solve_sylvester(transition_inv, -matrix, transition_inv @ initial_vector)
    g.ndata['pos_enc'] = torch.from_numpy(pe.T).float()
    return g

def add_automaton_encodings(dataset, transition_matrix, initial_vector):
    # Graph positional encoding w/ pre-computed automaton encoding
    dataset.train.graph_lists = [automaton_encoding(g, transition_matrix, initial_vector) for g in dataset.train.graph_lists]
    dataset.val.graph_lists = [automaton_encoding(g, transition_matrix, initial_vector) for g in dataset.val.graph_lists]
    dataset.test.graph_lists = [automaton_encoding(g, transition_matrix, initial_vector) for g in dataset.test.graph_lists]
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