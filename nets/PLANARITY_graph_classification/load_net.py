"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.PLANARITY_graph_classification.gat_net import GATNet
from nets.PLANARITY_graph_classification.gated_gcn_net import GatedGCNNet
from nets.PLANARITY_graph_classification.gin_net import GINNet
from nets.PLANARITY_graph_classification.graph_transformer import GraphTransformerNet
from nets.PLANARITY_graph_classification.graphsage_net import GraphSageNet

def GatedGCN(net_params):
    return GatedGCNNet(net_params)

def GIN(net_params):
    return GINNet(net_params)

def GraphTransformer(net_params):
    return GraphTransformerNet(net_params)

def GAT(net_params):
    return GATNet(net_params)

def GraphSAGE(net_params):
    return GraphSageNet(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'GatedGCN': GatedGCN,
        'GIN': GIN,
        'GraphTransformer': GraphTransformer,
        'GAT': GAT,
        'GraphSage': GraphSAGE
    }
        
    return models[MODEL_NAME](net_params)