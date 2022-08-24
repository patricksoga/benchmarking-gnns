from random import shuffle
import torch
import pickle
import torch.utils.data
import time
import os
import numpy as np

import dgl
import networkx as nx

from scipy import sparse as sp
import numpy as np
import torch.nn.functional as F
import networkx as nx

"""
    Part of this file is adapted from
    https://github.com/cvignac/SMP
"""


class K3ColorableDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, split):
        self.data_dir = data_dir
        self.split = split
        data = torch.load(os.path.join(self.data_dir, "k3colorable.pkl"))
        shuffle(data)
        self.data = data

        self.graph_lists = []
        self.graph_labels = []
        self._prepare()
    
    def get_proportions(self, data, split=(0.7, 0.15, 0.15)):
        n = len(data)
        t, v, te = split
        n_train = int(n*t)
        n_val =  int(n*v)
        n_test = int(n*te)
        return n_train, n_val, n_test

    def _prepare(self):
        graphs = [d[0] for d in self.data]
        labels = [d[1] for d in self.data]

        n_train, n_val, n_test = self.get_proportions(graphs)
        if self.split == "train":
            n_graphs = n_train
            lower_bound = 0
            upper_bound = n_graphs
        if self.split == "val":
            n_graphs = n_val
            lower_bound = n_train
            upper_bound = lower_bound + n_val
        if self.split == "test":
            n_graphs = n_test
            lower_bound = n_train + n_val
            upper_bound = lower_bound + n_test

        self.graph_lists = graphs[lower_bound:upper_bound]
        self.graph_labels = labels[lower_bound:upper_bound]

        print("preparing %d graphs for the %s set..." % (n_graphs, self.split.upper()))

        del self.data
        
    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graph_lists)

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]
    
    
class K3ColorableDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, name='K3Colorable'):
        t0 = time.time()
        self.name = name
        data_dir = './data/k3colorable'
        
        self.train = K3ColorableDGL(data_dir, 'train')
        self.val = K3ColorableDGL(data_dir, 'val')
        self.test = K3ColorableDGL(data_dir, 'test')
        print("Time taken: {:.4f}s".format(time.time()-t0))
        


        
def positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    g.ndata['pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 

    # # Eigenvectors with scipy
    # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    # EigVec = EigVec[:, EigVal.argsort()] # increasing order
    # g.ndata['pos_enc'] = torch.from_numpy(np.abs(EigVec[:,1:pos_enc_dim+1])).float() 
    n = g.number_of_nodes()
    if n <= pos_enc_dim:
        g.ndata['pos_enc'] = F.pad(g.ndata['pos_enc'], (0, pos_enc_dim - n + 1), value=float('0'))

    return g


def make_full_graph(g):
    """
        Converting the given graph to fully connected
        This function just makes full connections
        removes available edge features 
    """

    full_g = dgl.from_networkx(nx.complete_graph(g.number_of_nodes()))
    full_g.ndata['feat'] = g.ndata['feat']
    full_g.edata['feat'] = torch.zeros(full_g.number_of_edges()).long()
    
    try:
        full_g.ndata['pos_enc'] = g.ndata['pos_enc']
    except:
        pass

    try:
        full_g.ndata['wl_pos_enc'] = g.ndata['wl_pos_enc']
    except:
        pass    
    
    return full_g


class K3ColorableDataset(torch.utils.data.Dataset):

    def __init__(self, name):
        """
            Loading Cycles datasets
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        data_dir = 'data/k3colorable/'
        try:
            with open(data_dir+name+'.pkl',"rb") as f:
                f = pickle.load(f)
                self.train = f[0]
                self.val = f[1]
                self.test = f[2]
            print('train, test, val sizes :',len(self.train),len(self.test),len(self.val))
            print("[I] Finished loading.")
        except FileNotFoundError:
            print("[E] Data pkl files not found. Please prepare the pkl files first.")
            
        print("[I] Data load time: {:.4f}s".format(time.time()-start))


    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        # labels = torch.tensor(np.array(labels))
        labels = torch.tensor(labels)
        batched_graph = dgl.batch(graphs)
        return batched_graph, labels      


    def _add_positional_encodings(self, pos_enc_dim):
        
        # Graph positional encoding v/ Laplacian eigenvectors
        self.train.graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.train.graph_lists]
        self.val.graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.val.graph_lists]
        self.test.graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.test.graph_lists]

    def _make_full_graph(self):
        
        # function for converting graphs to full graphs
        # this function will be called only if full_graph flag is True
        self.train.graph_lists = [make_full_graph(g) for g in self.train.graph_lists]
        self.val.graph_lists = [make_full_graph(g) for g in self.val.graph_lists]
        self.test.graph_lists = [make_full_graph(g) for g in self.test.graph_lists]

class DGLFormDataset(torch.utils.data.Dataset):
    """
        DGLFormDataset wrapping graph list and label list as per pytorch Dataset.
        *lists (list): lists of 'graphs' and 'labels' with same len().
    """
    def __init__(self, *lists):
        assert all(len(lists[0]) == len(li) for li in lists)
        self.lists = lists
        self.graph_lists = lists[0]
        self.graph_labels = lists[1]

    def __getitem__(self, index):
        return tuple(li[index] for li in self.lists)

    def __len__(self):
        return len(self.lists[0])
    