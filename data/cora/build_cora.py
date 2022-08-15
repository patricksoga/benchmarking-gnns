import dgl.data.citation_graph
import dgl.data
import numpy as np
import torch


def main():
    masks = {'train_masks': [], 'val_masks': [], 'test_masks': []}
    # for i in range(10):
    #     np.random.seed(np.random.randint(0, 1000))

    #     dataset = dgl.data.CoraGraphDataset(reverse_edge=False, force_reload=True)
    #     dgl.data.utils.add_nodepred_split(dataset, [0.7, 0.15, 0.15])

    #     masks['train_masks'].append(dataset[0].ndata['train_mask'])
    #     masks['val_masks'].append(dataset[0].ndata['val_mask'])
    #     masks['test_masks'].append(dataset[0].ndata['test_mask'])
    for i in range(10):
        with np.load(f'./cora_split_0.6_0.2_{i}.npz') as splits_file:
            train_mask = splits_file['train_mask']
            val_mask = splits_file['val_mask']
            test_mask = splits_file['test_mask']

        masks['train_masks'].append(train_mask)
        masks['val_masks'].append(val_mask)
        masks['test_masks'].append(test_mask)

    torch.save(masks, f'./split_masks.pt')

if __name__ == '__main__':
    main()