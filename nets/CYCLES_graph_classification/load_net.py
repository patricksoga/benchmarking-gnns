"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.CYCLES_graph_classification.gated_gcn_net import GatedGCNNet
from nets.CYCLES_graph_classification.gin_net import GINNet
from nets.CYCLES_graph_classification.graph_transformer import GraphTransformerNet
from nets.CYCLES_graph_classification.sa_graph_transformer import SAGraphTransformerNet

def GatedGCN(net_params):
    return GatedGCNNet(net_params)

def GIN(net_params):
    return GINNet(net_params)

def GraphTransformer(net_params):
    return GraphTransformerNet(net_params)

def SAGraphTransformer(net_params):
    return SAGraphTransformerNet(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'GatedGCN': GatedGCN,
        'GIN': GIN,
        'GraphTransformer': GraphTransformer,
        'SAGraphTransformer': SAGraphTransformer
    }
        
    return models[MODEL_NAME](net_params)