"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.cora_node_classification.gated_gcn_net import GatedGCNNet
from nets.cora_node_classification.graph_transformer import GraphTransformerNet
from nets.cora_node_classification.gat_net import GATNet

def GatedGCN(net_params):
    return GatedGCNNet(net_params)

def GraphTransformer(net_params):
    return GraphTransformerNet(net_params)

def GAT(net_params):
    return GATNet(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'GatedGCN': GatedGCN,
        'GraphTransformer': GraphTransformer,
        'GAT': GAT
    }
        
    return models[MODEL_NAME](net_params)
