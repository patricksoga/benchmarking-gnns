import dgl.data
import numpy as np
import torch


def main():
    masks = {'train_masks': [], 'val_masks': [], 'test_masks': []}

    for i in range(10):
        with np.load(f'./wisconsin_split_0.6_0.2_{i}.npz') as splits_file:
            train_mask = splits_file['train_mask']
            val_mask = splits_file['val_mask']
            test_mask = splits_file['test_mask']

        masks['train_masks'].append(train_mask)
        masks['val_masks'].append(val_mask)
        masks['test_masks'].append(test_mask)

    torch.save(masks, f'./split_masks.pt')

if __name__ == '__main__':
    main()