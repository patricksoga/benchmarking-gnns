import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SpectralAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        embed_dim = config['spectral_embed_dim']
        lpe_n_heads = config['lpe_n_heads']
        lpe_n_layers = config['lpe_n_layers']

        encoder_layer = nn.TransformerEncoderLayer(embed_dim, lpe_n_heads)
        self.lpe_attn = nn.TransformerEncoder(encoder_layer, lpe_n_layers)
        # self.lpe_attn = nn.TransformerEncoderLayer(lpe_n_layers, lpe_n_heads, embed_dim, lpe_ff_dim)
        self.linear = nn.Linear(2, embed_dim)

    def forward(self, h, eigvecs, eigvals):

        lpe = torch.cat((eigvecs, eigvals), dim=2) # (Num nodes) x (Num Eigenvectors) x 2
        empty_mask = torch.isnan(lpe) # (Num nodes) x (Num Eigenvectors) x 2

        lpe[empty_mask] = 0 # (Num nodes) x (Num Eigenvectors) x 2
        lpe = torch.transpose(lpe, 0 ,1) # (Num Eigenvectors) x (Num nodes) x 2
        lpe = self.linear(lpe) # (Num Eigenvectors) x (Num nodes) x PE_dim

        lpe = self.lpe_attn(lpe, empty_mask[:,:,0])

        #remove masked sequences
        lpe[torch.transpose(empty_mask, 0 ,1)[:,:,0]] = float('nan')

        #Sum pooling
        lpe = torch.nansum(lpe, 0, keepdim=False)

        h = torch.cat((h, lpe), dim=1)
        return h