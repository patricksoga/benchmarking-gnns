import os
import os.path as osp
import shutil
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download, download_url, extract_zip
import numpy as np
from dgl.data.utils import load_graphs, Subset
from typing import Dict
import dgl
import torch

class DglPCQM4Mv2Dataset(object):
    def __init__(self, root = 'dataset'):
        '''
        DGL PCQM4Mv2 dataset object
            - root (str): the dataset folder will be located at root/pcqm4m_kddcup2021
        '''

        self.original_root = root
        self.folder = osp.join(root, 'pcqm4m-v2')
        self.version = 1

        # Old url hosted at Stanford
        # md5sum: 65b742bafca5670be4497499db7d361b
        # self.url = f'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2.zip'
        # New url hosted by DGL team at AWS--much faster to download
        self.url = 'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip'

        # check version and update if necessary
        if osp.isdir(self.folder) and (not osp.exists(osp.join(self.folder, f'RELEASE_v{self.version}.txt'))):
            print('PCQM4Mv2 dataset has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.folder)

        super(DglPCQM4Mv2Dataset, self).__init__()

        # Prepare everything.
        # download if there is no raw file
        # preprocess if there is no processed file
        # load data if processed file is found.
        self.prepare_graph()

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
        else:
            print('Stop download.')
            exit(-1)

    def prepare_graph(self):
        processed_dir = osp.join(self.folder, 'processed')
        raw_dir = osp.join(self.folder, 'raw')
        pre_processed_file_path = osp.join(processed_dir, 'dgl_data_processed')

        if osp.exists(pre_processed_file_path):        
            # if pre-processed file already exists
            self.graphs, label_dict = load_graphs(pre_processed_file_path)
            self.labels = label_dict['labels']

    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(torch.load(osp.join(self.folder, 'split_dict.pt')))
        return split_dict

    def __getitem__(self, idx):
        '''Get datapoint with index'''

        if isinstance(idx, int):
            return self.graphs[idx], self.labels[idx]
        elif torch.is_tensor(idx) and idx.dtype == torch.long:
            if idx.dim() == 0:
                return self.graphs[idx], self.labels[idx]
            elif idx.dim() == 1:
                return Subset(self, idx.cpu())

        raise IndexError(
            'Only integers and long are valid '
            'indices (got {}).'.format(type(idx).__name__))

    def __len__(self):
        '''Length of the dataset
        Returns
        -------
        int
            Length of Dataset
        '''
        return len(self.graphs)

    def __repr__(self):  # pragma: no cover
        return '{}({})'.format(self.__class__.__name__, len(self))

# Collate function for ordinary graph classification 
def collate_dgl(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    if isinstance(labels[0], torch.Tensor):
        return batched_graph, torch.stack(labels)
    else:
        return batched_graph, labels

class PCQM4Mv2Evaluator:
    def __init__(self):
        '''
            Evaluator for the PCQM4Mv2 dataset
            Metric is Mean Absolute Error
        '''
        pass 

    def eval(self, input_dict):
        '''
            y_true: numpy.ndarray or torch.Tensor of shape (num_graphs,)
            y_pred: numpy.ndarray or torch.Tensor of shape (num_graphs,)
            y_true and y_pred need to be of the same type (either numpy.ndarray or torch.Tensor)
        '''
        assert('y_pred' in input_dict)
        assert('y_true' in input_dict)

        y_pred, y_true = input_dict['y_pred'], input_dict['y_true']

        assert((isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray))
                or
                (isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor)))
        assert(y_true.shape == y_pred.shape)
        assert(len(y_true.shape) == 1)

        if isinstance(y_true, torch.Tensor):
            return {'mae': torch.mean(torch.abs(y_pred - y_true)).cpu().item()}
        else:
            return {'mae': float(np.mean(np.absolute(y_pred - y_true)))}

    def save_test_submission(self, input_dict: Dict, dir_path: str, mode: str):
        '''
            save test submission file at dir_path
        '''
        assert('y_pred' in input_dict)
        assert mode in ['test-dev', 'test-challenge']

        y_pred = input_dict['y_pred']

        if mode == 'test-dev':
            filename = osp.join(dir_path, 'y_pred_pcqm4m-v2_test-dev')
            assert(y_pred.shape == (147037,))
        elif mode == 'test-challenge':
            filename = osp.join(dir_path, 'y_pred_pcqm4m-v2_test-challenge')
            assert(y_pred.shape == (147432,))

        assert(isinstance(filename, str))
        assert(isinstance(y_pred, np.ndarray) or isinstance(y_pred, torch.Tensor))

        if not osp.exists(dir_path):
            os.makedirs(dir_path)

        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.numpy()
        y_pred = y_pred.astype(np.float32)
        np.savez_compressed(filename, y_pred = y_pred)

if __name__ == '__main__':
    dataset = DglPCQM4Mv2Dataset()
    print(dataset)
    print(dataset[100])
    split_dict = dataset.get_idx_split()
    print(split_dict)
    print(dataset[split_dict['train']])
    print(collate_dgl([dataset[0], dataset[1], dataset[2]]))
