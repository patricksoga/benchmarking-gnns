from pprint import pprint
import time
import dgl
import torch
from torch.utils.data import Dataset


from scipy import sparse as sp
import numpy as np

from .webkb_utils import full_load_data


def positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """
    
    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # # Eigenvectors with numpy
    # EigVal, EigVec = np.linalg.eig(L.toarray())
    # idx = EigVal.argsort() # increasing order
    # EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    # g.ndata['pos_enc'] = torch.from_numpy(np.abs(EigVec[:,1:pos_enc_dim+1])).float() 

    # Eigenvectors with scipy
    #EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2)
    EigVec = EigVec[:, EigVal.argsort()] # increasing order
    g.ndata['pos_enc'] = torch.from_numpy(np.real(EigVec[:,1:pos_enc_dim+1])).float() 

    return g


class WebKBDataset(Dataset):
    def __init__(self, name):
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        self.graph, features, labels, num_features, num_labels = full_load_data(name)

        masks = torch.load(f'./data/{name}/split_masks.pt')
        self.train_masks = [torch.BoolTensor(mask) for mask in masks['train_masks']]
        self.val_masks = [torch.BoolTensor(mask) for mask in masks['val_masks']]
        self.test_masks = [torch.BoolTensor(mask) for mask in masks['test_masks']]

        self.n_features = num_features
        self.n_classes = self.graph.ndata['label'].max().item() + 1
        self.graph.edata['feat'] = torch.zeros(self.graph.number_of_edges(), 1)

        self.labels = self.graph.ndata['label']

        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))

    def _add_positional_encodings(self, pos_enc_dim):
        # Graph positional encoding v/ Laplacian eigenvectors
        self.graph = positional_encoding(self.graph, pos_enc_dim)
